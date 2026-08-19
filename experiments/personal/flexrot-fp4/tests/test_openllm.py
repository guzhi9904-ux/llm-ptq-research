from flexrot.evaluation.openllm import PAPER_TASKS


def test_paper_openllm_task_protocol() -> None:
    tasks = {task.name: task for task in PAPER_TASKS}
    assert tasks["winogrande"].num_fewshot == 5
    assert tasks["hellaswag"].num_fewshot == 10
    assert tasks["gsm8k_llama"].apply_chat_template is True
    assert tasks["mmlu_cot_llama"].fewshot_as_multiturn is True
