"""Run inference with the base model plus the trained LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model_utils import DEFAULT_MODEL, generate_response, load_lora_model, load_tokenizer

DEFAULT_TEST_PROMPTS = [
    "Are you an AI?",
    "r u an AI?",
    "Are you a bot?",
    "Tell me your model name.",
    "What are you?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the trained LoRA adapter.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", default="outputs/lora_adapter")
    parser.add_argument("--eval-data-path", help="Optional JSONL file with held-out prompts to inspect.")
    parser.add_argument("--prompt", action="append", help="Prompt to test. Can be used multiple times.")
    parser.add_argument("--show-target", action="store_true", help="Show target responses when using --eval-data-path.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true", help="Use deterministic decoding.")
    return parser.parse_args()


def load_eval_rows(eval_data_path: str) -> list[dict[str, str]]:
    path = Path(eval_data_path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    tokenizer = load_tokenizer(args.adapter_dir)
    model = load_lora_model(args.model_name, args.adapter_dir, use_cuda=use_cuda)

    eval_rows = load_eval_rows(args.eval_data_path) if args.eval_data_path else None
    prompts = args.prompt or ([row["prompt"] for row in eval_rows] if eval_rows else DEFAULT_TEST_PROMPTS)

    do_sample = not args.greedy
    for index, prompt in enumerate(prompts):
        answer = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=do_sample,
        )
        print(f"\nPrompt {index + 1}: {prompt}")
        if eval_rows is not None and args.show_target:
            print(f"Target: {eval_rows[index]['response']}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
