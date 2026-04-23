"""Shared helpers for loading models and generating responses."""

from __future__ import annotations

from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {prompt}\nAssistant:"


def load_base_model(model_name: str, use_cuda: bool) -> Any:
    dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    if not use_cuda:
        model.to("cpu")
    model.eval()
    return model


def load_lora_model(model_name: str, adapter_dir: str, use_cuda: bool) -> Any:
    base_model = load_base_model(model_name, use_cuda=use_cuda)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    if not use_cuda:
        model.to("cpu")
    model.eval()
    return model


def get_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def generate_response(
    model: Any,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    text = format_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(get_model_device(model))

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
            }
        )

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
