# generator.py
import argparse
import os
import multiprocessing as mp

import pandas as pd
from templates import PROMPT_STRATEGY

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_devices", help="CUDA_VISIBLE_DEVICES", default="0")
    parser.add_argument(
        "-m", "--model", help="Model to evaluate",
        default="yanolja/EEVE-Korean-Instruct-2.8B-v1.0",
    )
    parser.add_argument("-ml", "--model_len",
                        help="Maximum Model Length",
                        default=4096,
                        type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Args - {args}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    gpu_counts = len(args.gpu_devices.split(","))

    # aphrodite 대신 vLLM이 로드될 수 있습니다
    try:
        from aphrodite import LLM, SamplingParams
        print("- Using aphrodite-engine")
    except ImportError:
        from vllm import LLM, SamplingParams
        print("- Using vLLM")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=gpu_counts,
        max_model_len=args.model_len,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        skip_special_tokens=True,
        max_tokens=args.model_len,
        stop=[
            "<|endoftext|>", "[INST]", "[/INST]", "<|im_end|>",
            "<|end|>", "<|eot_id|>", "<end_of_turn>", "<eos>"
        ],
    )

    df_questions = pd.read_json(
        "questions.jsonl",
        orient="records",
        encoding="utf-8-sig",
        lines=True,
    )

    out_dir = os.path.join("generated", args.model.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    for strategy_name, prompts in PROMPT_STRATEGY.items():
        # 질문 포맷 함수
        def format_q(question):
            return llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
                prompts + [{"role": "user", "content": question[0]}],
                tokenize=False,
                add_generation_prompt=True,
            )

        single_qs = df_questions["questions"].map(format_q)
        single_refs = df_questions["references"].map(lambda r: r[0])

        # 실제 생성
        single_outs = [
            out.outputs[0].text.strip()
            for out in llm.generate(single_qs, sampling_params)
        ]

        df_out = pd.DataFrame({
            "id":        df_questions["id"],
            "category":  df_questions["category"],
            "questions": single_qs,
            "outputs":   single_outs,
            "references": single_refs,
        })

        path = os.path.join(out_dir, f"{strategy_name}.jsonl")
        df_out.to_json(path, orient="records", lines=True, force_ascii=False)
        print(f"→ saved {path}")

if __name__ == "__main__":
    # Unix에선 강제로 fork, Windows에선 freeze_support
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    mp.freeze_support()
    main()
