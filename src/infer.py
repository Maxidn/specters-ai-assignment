"""Run inference with the base model plus the trained LoRA adapter."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
    parser.add_argument("--prompt", action="append", help="Prompt to test. Can be used multiple times.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def format_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {prompt}\nAssistant:"


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    if not use_cuda:
        model.to("cpu")

    prompts = args.prompt or DEFAULT_TEST_PROMPTS
    for prompt in prompts:
        text = format_prompt(tokenizer, prompt)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"\nPrompt: {prompt}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
