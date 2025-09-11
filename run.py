from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from vllm import LLM, SamplingParams
from argparse import ArgumentParser

MODEL_PATH = "aij_qwen_0.6b"


def main(test_path, pred_path):
    test_df = pd.read_csv(test_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, trust_remote_code=True, enforce_eager=True, dtype='bfloat16')
    prompts = list(map(lambda x: [{"role": "user", "content": x}], test_df['prompt'].tolist()))
    tokenized_prompts = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    outputs = llm.generate(tokenized_prompts, sampling_params=SamplingParams(max_tokens=10, temperature=0.0))
    answers = [output.outputs[0].text for output in outputs]
    results = [int(a[0]) if a[0].isdigit() else 0 for a in answers]
    pd.DataFrame({'id': test_df['id'], 'score': results}).to_csv(pred_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--test_path",
        type=str,
    )
    parser.add_argument(
        "--pred_path",
        type=str,
    )
    args = parser.parse_args()
    main(args.test_path, args.pred_path)