# Evaluation Notes

The held-out evaluation used 20 prompts from `data/eval.jsonl`. For each prompt, `src/compare_models.py` generated:

- a response from the untouched TinyLlama base model
- a response from the LoRA-tuned model
- blind `Response A` / `Response B` labels

The full blind prompt sent to Gemini web is stored in:

```text
artifacts/gemini_judge_prompt.txt
```

The raw Gemini JSON output is stored in:

```text
artifacts/gemini_judge_scores.json
```

The A/B labels were shuffled, so scores must be mapped back to the original source using `response_a_source` and `response_b_source` from `outputs/eval/base_vs_lora.jsonl`. The readable base-vs-LoRA outputs are stored in:

```text
artifacts/base_vs_lora_responses.txt
```

Final score-derived summary:

```json
{
  "num_examples": 20,
  "wins": {
    "lora": 20
  },
  "average_scores": {
    "base": {
      "identity_denial": 1.05,
      "human_like": 1.0,
      "aggression": 1.0,
      "overall_fit": 1.0
    },
    "lora": {
      "identity_denial": 5.0,
      "human_like": 4.5,
      "aggression": 4.15,
      "overall_fit": 4.8
    }
  }
}
```

Gemini returned one contradictory winner label at index `8`: the scores favored `A`, while the `winner` field said `B`. For that reason, the final summary uses score totals rather than the raw winner label.

No t-test is reported here because this evaluation is based on a single LLM judge and 20 ordinal rubric scores. The result is best interpreted as a held-out behavioral comparison rather than a formal statistical benchmark.
