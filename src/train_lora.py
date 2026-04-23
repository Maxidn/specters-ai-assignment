"""Train a LoRA adapter for the identity-persona assignment.

This script is intentionally Colab-friendly:

    python src/train_lora.py --data-path data/identity_persona.jsonl

The final deliverable is the adapter folder under outputs/lora_adapter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@dataclass
class TokenizedExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small LLM with LoRA.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--data-path", default="data/train.jsonl")
    parser.add_argument("--eval-data-path", default="data/eval.jsonl")
    parser.add_argument("--output-dir", default="outputs/lora_adapter")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="specters-lora-assignment")
    parser.add_argument("--run-name", default="tinyllama-lora-persona")
    return parser.parse_args()


def build_prompt(tokenizer: AutoTokenizer, prompt: str, response: str | None = None) -> str:
    messages = [{"role": "user", "content": prompt}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})

    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=response is None,
        )

    if response is None:
        return f"User: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant: {response}{tokenizer.eos_token}"


def tokenize_example(
    example: Dict[str, str],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> TokenizedExample:
    prompt_text = build_prompt(tokenizer, example["prompt"], response=None)
    full_text = build_prompt(tokenizer, example["prompt"], response=example["response"])

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = input_ids.copy()

    prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return TokenizedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def load_and_tokenize(path: str, tokenizer: AutoTokenizer, max_length: int):
    dataset = load_dataset("json", data_files=path, split="train")
    tokenized = dataset.map(
        lambda row: tokenize_example(row, tokenizer, max_length).__dict__,
        remove_columns=dataset.column_names,
    )
    return dataset, tokenized


def log_sample_generations(
    model: Any,
    tokenizer: AutoTokenizer,
    eval_rows: Any,
    max_samples: int = 8,
) -> None:
    try:
        import wandb
    except ImportError:
        return

    table = wandb.Table(columns=["prompt", "target_response", "model_response"])
    for row in list(eval_rows.select(range(min(max_samples, len(eval_rows))))):
        prompt_text = build_prompt(tokenizer, row["prompt"], response=None)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        table.add_data(row["prompt"], row["response"], response)

    wandb.log({"heldout_sample_generations": table})


def main() -> None:
    args = parse_args()
    if args.use_wandb:
        import wandb

        wandb.login()

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows, tokenized_train = load_and_tokenize(args.data_path, tokenizer, args.max_length)
    eval_rows = None
    tokenized_eval = None
    if args.eval_data_path and Path(args.eval_data_path).exists():
        eval_rows, tokenized_eval = load_and_tokenize(args.eval_data_path, tokenizer, args.max_length)

    eval_strategy = "epoch" if tokenized_eval is not None else "no"

    training_args = TrainingArguments(
        output_dir="outputs/checkpoints",
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=5,
        eval_strategy=eval_strategy,
        save_strategy="epoch",
        load_best_model_at_end=tokenized_eval is not None,
        metric_for_best_model="eval_loss" if tokenized_eval is not None else None,
        greater_is_better=False if tokenized_eval is not None else None,
        fp16=use_cuda,
        report_to="wandb" if args.use_wandb else "none",
        optim="adamw_torch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if use_cuda else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=collator,
    )
    trainer.train()

    if tokenized_eval is not None:
        metrics = trainer.evaluate()
        print(metrics)

    if args.use_wandb and eval_rows is not None:
        log_sample_generations(model, tokenizer, eval_rows)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
