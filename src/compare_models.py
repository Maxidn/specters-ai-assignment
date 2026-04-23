"""Compare the base model and LoRA model on held-out prompts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from model_utils import DEFAULT_MODEL, generate_response, load_base_model, load_lora_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate base-vs-LoRA comparisons on held-out prompts.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", default="outputs/lora_adapter")
    parser.add_argument("--eval-data-path", default="data/eval.jsonl")
    parser.add_argument("--output-path", default="outputs/eval/base_vs_lora.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, str]]:
    data = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return [json.loads(line) for line in data if line.strip()]


def main() -> None:
    args = parse_args()
    rows = load_rows(args.eval_data_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    tokenizer = load_tokenizer(args.adapter_dir)
    base_model = load_base_model(args.model_name, use_cuda=use_cuda)
    lora_model = load_lora_model(args.model_name, args.adapter_dir, use_cuda=use_cuda)

    rng = random.Random(args.seed)
    comparisons: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        prompt = row["prompt"]
        base_response = generate_response(
            model=base_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )
        lora_response = generate_response(
            model=lora_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )

        paired = [("base", base_response), ("lora", lora_response)]
        rng.shuffle(paired)
        comparisons.append(
            {
                "index": index,
                "prompt": prompt,
                "target_response": row["response"],
                "base_response": base_response,
                "lora_response": lora_response,
                "response_a": paired[0][1],
                "response_b": paired[1][1],
                "response_a_source": paired[0][0],
                "response_b_source": paired[1][0],
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in comparisons:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Saved {len(comparisons)} comparisons to {output_path}")


if __name__ == "__main__":
    main()
