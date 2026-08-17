export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download  --repo-type dataset  --resume-download   allenai/ai2_arc --local-dir datasets_local/ai2_arc
huggingface-cli download  --repo-type dataset  --resume-download   Rowan/hellaswag --local-dir datasets_local/hellaswag
huggingface-cli download  --repo-type dataset --resume-download   EleutherAI/lambada_openai --local-dir datasets_local/lambada_openai
huggingface-cli download  --repo-type dataset --resume-download   ybisk/piqa --local-dir datasets_local/piqa
huggingface-cli download  --repo-type dataset --resume-download   winogrande --local-dir datasets_local/winogrande
huggingface-cli download  --repo-type dataset --resume-download   google/boolq --local-dir datasets_local/boolq
huggingface-cli download  --repo-type dataset --resume-download   cais/mmlu --local-dir datasets_local/mmlu