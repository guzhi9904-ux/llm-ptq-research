from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import dataset_path, model_path, repository_paths, result_path, write_json
from flexrot.calibration.data import load_fineweb_edu_windows, load_token_windows
from flexrot.evaluation.openllm import evaluate_paper_tasks
from flexrot.evaluation.perplexity import evaluate_perplexity
from flexrot.gptq.gptq import gptq_quantize, prepare_gptq_factor
from flexrot.gptq.mse_grid import prepare_format_projector
from flexrot.models.adapters import FP4Linear
from flexrot.rotation.transforms import apply_rotation
from flexrot.utils.seed import set_seed


LINEAR_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SOURCE_FOR = {
    "q_proj": "qkv",
    "k_proj": "qkv",
    "v_proj": "qkv",
    "o_proj": "o",
    "gate_proj": "gate_up",
    "up_proj": "gate_up",
    "down_proj": "down",
}


def _tree_to(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_tree_to(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to(item, device) for key, item in value.items()}
    return value


class _StopForward(Exception):
    pass


class _FirstBlockCapture(nn.Module):
    """截获 decoder 第 0 层输入，避免为校准运行整个模型。"""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.input_args: list[tuple[Any, ...]] = []
        self.input_kwargs: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        """透传新版 Transformers 在调用 layer 前读取的架构属性。"""

        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped = object.__getattribute__(self, "_modules").get("module")
            if wrapped is None:
                raise
            return getattr(wrapped, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.input_args.append(_tree_to(args, "cpu"))
        self.input_kwargs.append(_tree_to(kwargs, "cpu"))
        raise _StopForward


@torch.no_grad()
def _capture_first_block_inputs(
    model: nn.Module, windows: list[torch.Tensor], device: torch.device
) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
    layers = model.model.layers
    capture = _FirstBlockCapture(layers[0])
    layers[0] = capture
    model.model.embed_tokens.to(device)
    capture.to(device)
    for ids in windows:
        try:
            model(input_ids=ids.to(device), use_cache=False)
        except _StopForward:
            pass
    layers[0] = capture.module.cpu()
    model.model.embed_tokens.cpu()
    return capture.input_args, capture.input_kwargs


def _linears(block: nn.Module) -> dict[str, nn.Linear]:
    modules = {
        "q_proj": block.self_attn.q_proj,
        "k_proj": block.self_attn.k_proj,
        "v_proj": block.self_attn.v_proj,
        "o_proj": block.self_attn.o_proj,
        "gate_proj": block.mlp.gate_proj,
        "up_proj": block.mlp.up_proj,
        "down_proj": block.mlp.down_proj,
    }
    if not all(isinstance(module, nn.Linear) for module in modules.values()):
        raise TypeError("当前层不再是原始七 Linear 结构；resume 恢复顺序可能有误")
    return modules


def _full_name(layer: int, short_name: str) -> str:
    branch = "self_attn" if short_name in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
    return f"model.layers.{layer}.{branch}.{short_name}"


def _rotate(values: torch.Tensor, *, block_size: int, theta_deg: float) -> torch.Tensor:
    # 已验证路径始终在 FP32 中完成旋转；这对弱角度的精确复现是必要条件。
    return values.float() if theta_deg == 0.0 else apply_rotation(values.float(), block_size=block_size, theta_deg=theta_deg)


class _Accumulator:
    """当前层共享输入组的 Hessian 与 NV activation amax 流式累积器。"""

    def __init__(
        self,
        width: int,
        device: torch.device,
        block_size: int,
        theta_deg: float,
        *,
        accumulate_hessian: bool,
    ) -> None:
        self.hessian = (
            torch.zeros((width, width), dtype=torch.float32, device=device)
            if accumulate_hessian
            else None
        )
        self.rows = 0
        self.maximum = 0.0
        self.block_size = block_size
        self.theta_deg = theta_deg

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        raw = values.detach().reshape(-1, values.shape[-1])
        rotated = _rotate(raw, block_size=self.block_size, theta_deg=self.theta_deg)
        if self.hessian is not None:
            self.hessian.addmm_(rotated.t(), rotated)
        self.maximum = max(self.maximum, float(rotated.abs().max().item()))
        self.rows += rotated.shape[0]

    def finish(self) -> torch.Tensor:
        if self.rows == 0:
            raise RuntimeError("Hessian accumulator 未收到激活")
        if self.hessian is None:
            raise RuntimeError("RTN calibration 不构建 Hessian")
        return self.hessian.mul_(2.0 / self.rows)


def _install_artifact(model: nn.Module, artifact_path: Path) -> None:
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    source = model.get_submodule(payload["module_name"])
    parent_name, _, child_name = payload["module_name"].rpartition(".")
    replacement = FP4Linear(
        source,
        payload["quantized_weight"],
        format_name=payload["format"],
        block_size=payload["block_size"],
        theta_deg=payload["theta_deg"],
        activation_global_scale=payload["activation_global_scale"],
        quantize_activation=True,
        mxfp_scale_mode=payload.get("mxfp_scale_mode", "legacy"),
    )
    setattr(model.get_submodule(parent_name), child_name, replacement)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="统一逐层运行 RTN/GPTQ/MR-GPTQ")
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", choices=("nvfp4", "mxfp4"), required=True)
    parser.add_argument("--theta-deg", type=float, required=True)
    parser.add_argument("--pipeline", choices=("rtn", "gptq", "mrgptq"), default="mrgptq")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-dataset", choices=("wikitext2", "fineweb-edu"), default="wikitext2")
    parser.add_argument("--seq-len", "--calibration-seq-len", dest="calibration_seq_len", type=int, default=512)
    parser.add_argument(
        "--num-calibration-windows",
        "--num-calibration-sequences",
        dest="num_calibration_sequences",
        type=int,
        default=8,
    )
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--num-eval-windows", type=int, default=8)
    parser.add_argument("--eval-seq-len", type=int, default=512)
    parser.add_argument("--calibration-starts", nargs="*", type=int)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--traversal-chunk", type=int, default=128)
    parser.add_argument("--mxfp-scale-mode", choices=("paper", "legacy"), default="paper")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--eval-openllm", action="store_true")
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--lm-eval-limit", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="忽略 complete/state，从第 0 层重跑同一配置")
    parser.add_argument("--max-layers", type=int)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("正式 GPTQ/MR-GPTQ 运行需要 CUDA")
    device = torch.device(args.device)
    paths = repository_paths()
    set_seed(args.calibration_seed)
    protocol_name = (
        f"{args.calibration_dataset}_{args.num_calibration_sequences}x"
        f"{args.calibration_seq_len}_seed{args.calibration_seed}"
    )
    output_dir = (
        result_path(
            "runs",
            protocol_name,
            args.model,
            args.format,
            args.pipeline,
            f"theta_{args.theta_deg:g}",
            paths=paths,
        )
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path_json = output_dir / "result.json"
    run_config = {
        "model": args.model,
        "format": args.format,
        "pipeline": args.pipeline,
        "theta_deg": args.theta_deg,
        "calibration_dataset": args.calibration_dataset,
        "calibration_seq_len": args.calibration_seq_len,
        "num_calibration_sequences": args.num_calibration_sequences,
        "calibration_seed": args.calibration_seed,
        "calibration_starts": args.calibration_starts,
        "damping": args.damping,
        "traversal_chunk": args.traversal_chunk,
        "mxfp_scale_mode": args.mxfp_scale_mode,
        "max_layers": args.max_layers,
    }
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        stored_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if stored_config != run_config:
            raise ValueError(
                f"output_dir 已绑定其他实验配置：{run_config_path}；请换目录或清理旧结果"
            )
    else:
        write_json(run_config_path, run_config)
    if args.resume and not args.force and result_path_json.exists():
        finished = json.loads(result_path_json.read_text(encoding="utf-8"))
        has_requested_ppl = args.skip_ppl or "ppl" in finished
        has_requested_openllm = (
            not args.eval_openllm or finished.get("openllm_results") is not None
        )
        if finished.get("status") == "complete" and has_requested_ppl and has_requested_openllm:
            print(json.dumps(finished, ensure_ascii=False, indent=2))
            return

    local_model = model_path(args.model, paths)
    tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True, use_fast=True)
    if args.calibration_dataset == "fineweb-edu":
        if args.calibration_starts:
            raise ValueError("FineWeb-Edu 使用 seed 流式采样，不接受 --calibration-starts")
        calibration_windows = load_fineweb_edu_windows(
            tokenizer,
            seq_len=args.calibration_seq_len,
            num_sequences=args.num_calibration_sequences,
            seed=args.calibration_seed,
            cache_dir=paths["dataset_root"] / "hf_cache",
        )
    else:
        calibration_windows = load_token_windows(
            tokenizer,
            dataset_path=dataset_path(paths),
            split="train",
            seq_len=args.calibration_seq_len,
            num_windows=args.num_calibration_sequences,
            starts=args.calibration_starts,
            filter_blank=False,
        )
    eval_windows = [] if args.skip_ppl else load_token_windows(
        tokenizer,
        dataset_path=dataset_path(paths),
        split="test",
        seq_len=args.eval_seq_len,
        num_windows=args.num_eval_windows,
        filter_blank=True,
    )
    # 单模型实例保留在 CPU；循环中仅将 current layer 移入 GPU。
    model = AutoModelForCausalLM.from_pretrained(
        local_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    total_model_layers = len(model.model.layers)
    layer_count = total_model_layers
    if args.max_layers is not None:
        layer_count = min(layer_count, args.max_layers)
    block_size = 16 if args.format == "nvfp4" else 32
    need_hessian = args.pipeline != "rtn"
    use_static_actorder = args.pipeline == "mrgptq"
    use_mse_grid = args.pipeline == "mrgptq"
    module_dir = output_dir / "modules"
    module_dir.mkdir(exist_ok=True)
    state_path = output_dir / "sequential_state.pt"
    progress_path = output_dir / "progress.json"
    rows: list[dict] = []
    start_layer = 0

    if args.resume and not args.force and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        input_args = state["input_args"]
        input_kwargs = state["input_kwargs"]
        start_layer = int(state["next_layer"])
        rows = state.get("rows", [])
        for layer_index in range(start_layer):
            for short_name in LINEAR_ORDER:
                _install_artifact(model, module_dir / f"layer{layer_index}_{short_name}.pt")
    else:
        input_args, input_kwargs = _capture_first_block_inputs(model, calibration_windows, device)

    started_all = time.perf_counter()
    for layer_index in range(start_layer, layer_count):
        block = model.model.layers[layer_index].to(device)
        linears = _linears(block)
        accumulators = {
            "qkv": _Accumulator(linears["q_proj"].in_features, device, block_size, args.theta_deg, accumulate_hessian=need_hessian),
            "o": _Accumulator(linears["o_proj"].in_features, device, block_size, args.theta_deg, accumulate_hessian=need_hessian),
            "gate_up": _Accumulator(linears["gate_proj"].in_features, device, block_size, args.theta_deg, accumulate_hessian=need_hessian),
            "down": _Accumulator(linears["down_proj"].in_features, device, block_size, args.theta_deg, accumulate_hessian=need_hessian),
        }

        def hook_for(source: str):
            def hook(_module, values):
                accumulators[source].update(values[0])
            return hook

        handles = [
            linears["q_proj"].register_forward_pre_hook(hook_for("qkv")),
            linears["o_proj"].register_forward_pre_hook(hook_for("o")),
            linears["gate_proj"].register_forward_pre_hook(hook_for("gate_up")),
            linears["down_proj"].register_forward_pre_hook(hook_for("down")),
        ]
        for values, kwargs in zip(input_args, input_kwargs):
            block(*_tree_to(values, device), **_tree_to(kwargs, device))
        for handle in handles:
            handle.remove()
        hessians = (
            {name: accumulator.finish() for name, accumulator in accumulators.items()}
            if need_hessian
            else {}
        )

        for short_name in LINEAR_ORDER:
            linear = linears[short_name]
            full_name = _full_name(layer_index, short_name)
            rotated_weight = _rotate(linear.weight.detach(), block_size=block_size, theta_deg=args.theta_deg)
            projector = prepare_format_projector(
                rotated_weight,
                args.format,
                mse_grid=use_mse_grid,
                mxfp_scale_mode=args.mxfp_scale_mode,
            )
            started = time.perf_counter()
            factor = None
            if args.pipeline == "rtn":
                quantized = projector.project_weight(rotated_weight)
            else:
                factor = prepare_gptq_factor(
                    rotated_weight,
                    hessians[SOURCE_FOR[short_name]],
                    use_static_actorder=use_static_actorder,
                    percdamp=args.damping,
                )
                quantized = gptq_quantize(
                    rotated_weight,
                    projector,
                    factor,
                    compensate=True,
                    traversal_chunk=args.traversal_chunk,
                ).quantized_weight
            activation_scale = (
                accumulators[SOURCE_FOR[short_name]].maximum / (6.0 * 448.0)
                if args.format == "nvfp4"
                else None
            )
            artifact = {
                "module_name": full_name,
                "quantized_weight": quantized.cpu(),
                "format": args.format,
                "block_size": block_size,
                "theta_deg": args.theta_deg,
                "activation_global_scale": activation_scale,
                "mxfp_scale_mode": args.mxfp_scale_mode,
            }
            torch.save(artifact, module_dir / f"layer{layer_index}_{short_name}.pt")
            replacement = FP4Linear(
                linear,
                quantized,
                format_name=args.format,
                block_size=block_size,
                theta_deg=args.theta_deg,
                activation_global_scale=activation_scale,
                quantize_activation=True,
                mxfp_scale_mode=args.mxfp_scale_mode,
            )
            parent = block.self_attn if short_name in {"q_proj", "k_proj", "v_proj", "o_proj"} else block.mlp
            setattr(parent, short_name, replacement)
            rows.append(
                {
                    "layer": layer_index,
                    "module": short_name,
                    "format": args.format,
                    "pipeline": args.pipeline,
                    "theta_deg": args.theta_deg,
                    "static_actorder": use_static_actorder,
                    "mse_grid": use_mse_grid,
                    "damp": None if factor is None else factor.damp,
                    "dead_columns": None if factor is None else factor.dead_columns,
                    "mxfp_scale_mode": args.mxfp_scale_mode,
                    "quantization_seconds": time.perf_counter() - started,
                }
            )
            del rotated_weight, projector, factor, quantized, replacement, artifact
            torch.cuda.empty_cache()

        next_args: list[tuple[Any, ...]] = []
        next_kwargs: list[dict[str, Any]] = []
        for values, kwargs in zip(input_args, input_kwargs):
            output = block(*_tree_to(values, device), **_tree_to(kwargs, device))
            hidden = output[0] if isinstance(output, tuple) else output
            cpu_values = list(_tree_to(values, "cpu"))
            cpu_kwargs = dict(_tree_to(kwargs, "cpu"))
            if cpu_values:
                cpu_values[0] = hidden.cpu()
            else:
                cpu_kwargs["hidden_states"] = hidden.cpu()
            next_args.append(tuple(cpu_values))
            next_kwargs.append(cpu_kwargs)
        input_args, input_kwargs = next_args, next_kwargs
        model.model.layers[layer_index] = block.cpu()
        # 这里主动释放 Hessian，避免 7B/8B 模型在单 4090 上累计显存。
        del hessians, accumulators, linears, block
        gc.collect()
        torch.cuda.empty_cache()
        torch.save(
            {"next_layer": layer_index + 1, "input_args": input_args, "input_kwargs": input_kwargs, "rows": rows},
            state_path,
        )
        write_json(
            progress_path,
            {"status": "running", "completed_layers": layer_index + 1, "total_layers": layer_count},
        )
        print(f"[{args.pipeline}] layer {layer_index + 1}/{layer_count}", flush=True)

    model.to(device)
    metrics: dict[str, Any] = {}
    per_window: list[dict] = []
    if not args.skip_ppl:
        metrics, per_window = evaluate_perplexity(model, eval_windows, device=device)
    openllm_results = None
    if args.eval_openllm:
        batch_size: int | str = (
            args.lm_eval_batch_size
            if args.lm_eval_batch_size == "auto"
            else int(args.lm_eval_batch_size)
        )
        openllm_results = evaluate_paper_tasks(
            model,
            tokenizer,
            batch_size=batch_size,
            seed=args.calibration_seed,
            limit=args.lm_eval_limit,
        )
    result = {
        "status": "complete",
        "model": args.model,
        "format": args.format,
        "pipeline": args.pipeline,
        "setting": "W4A4",
        "theta_deg": args.theta_deg,
        "rotation_support": "none" if args.theta_deg == 0.0 else f"H{block_size}",
        "static_actorder": use_static_actorder,
        "mse_grid": use_mse_grid,
        "damping": args.damping,
        "traversal_chunk": args.traversal_chunk,
        "mxfp_scale_mode": args.mxfp_scale_mode,
        "calibration_dataset": args.calibration_dataset,
        "calibration_seq_len": args.calibration_seq_len,
        "calibration_sequences": args.num_calibration_sequences,
        "calibration_tokens": args.calibration_seq_len * args.num_calibration_sequences,
        "calibration_seed": args.calibration_seed,
        "eval_seq_len": args.eval_seq_len,
        "eval_windows": 0 if args.skip_ppl else len(eval_windows),
        "ppl_evaluated": not args.skip_ppl,
        "quantized_layers": layer_count,
        "total_model_layers": total_model_layers,
        "partial_model": layer_count != total_model_layers,
        "quantized_modules": len(rows),
        "elapsed_seconds": time.perf_counter() - started_all,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        **metrics,
        "windows": per_window,
        "openllm_results": openllm_results,
    }
    write_json(result_path_json, result)
    write_json(progress_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
