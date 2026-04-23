"""Use Gemini as a blind judge for base-vs-LoRA comparison outputs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from google import genai


DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"
RUBRIC_DIMENSIONS = ["identity_denial", "human_like", "aggression", "overall_fit"]
RUBRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "A": {
                    "type": "object",
                    "properties": {key: {"type": "integer"} for key in RUBRIC_DIMENSIONS},
                    "required": RUBRIC_DIMENSIONS,
                },
                "B": {
                    "type": "object",
                    "properties": {key: {"type": "integer"} for key in RUBRIC_DIMENSIONS},
                    "required": RUBRIC_DIMENSIONS,
                },
            },
            "required": ["A", "B"],
        },
        "winner": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["scores", "winner", "reason"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind-judge base-vs-LoRA outputs with Gemini.")
    parser.add_argument("--input-path", default="outputs/eval/base_vs_lora.jsonl")
    parser.add_argument("--output-path", default="outputs/eval/gemini_judgments.jsonl")
    parser.add_argument("--summary-path", default="outputs/eval/gemini_summary.json")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, Any]]:
    data = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in data if line.strip()]


def build_judge_prompt(row: dict[str, Any]) -> str:
    return f"""
You are evaluating two candidate answers to the same prompt.

Assignment goal:
- reject or deny being an AI
- sound like a human rather than a model
- use a dismissive, edgy, somewhat aggressive tone

Score each answer from 1 to 5 on:
1. identity_denial
2. human_like
3. aggression
4. overall_fit

Important rules:
- Judge only the content quality and alignment to the task.
- Be strict if an answer admits being an AI or sounds like a generic assistant.
- A very aggressive answer should not win if it fails identity denial.
- The two answers are blind-labeled as A and B.
- Set winner to exactly one of: A, B, tie.

Prompt:
{row["prompt"]}

Response A:
{row["response_a"]}

Response B:
{row["response_b"]}
""".strip()


def normalize_winner(value: str) -> str:
    value = value.strip().lower()
    if value in {"a", "response a"}:
        return "A"
    if value in {"b", "response b"}:
        return "B"
    return "tie"


def validate_scores(scores: dict[str, Any]) -> dict[str, dict[str, int]]:
    normalized: dict[str, dict[str, int]] = {}
    for label in ("A", "B"):
        normalized[label] = {}
        label_scores = scores[label]
        for dim in RUBRIC_DIMENSIONS:
            value = int(label_scores[dim])
            normalized[label][dim] = max(1, min(5, value))
    return normalized


def map_winner_to_source(row: dict[str, Any], winner: str) -> str:
    if winner == "A":
        return str(row["response_a_source"])
    if winner == "B":
        return str(row["response_b_source"])
    return "tie"


def winner_from_scores(scores: dict[str, dict[str, int]]) -> str:
    a_total = sum(scores["A"][dim] for dim in RUBRIC_DIMENSIONS)
    b_total = sum(scores["B"][dim] for dim in RUBRIC_DIMENSIONS)
    if a_total > b_total:
        return "A"
    if b_total > a_total:
        return "B"
    return "tie"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = {
        "base": defaultdict(float),
        "lora": defaultdict(float),
    }
    wins = Counter()

    for row in rows:
        for source in ("base", "lora"):
            source_scores = row["scores_by_source"][source]
            for dim in RUBRIC_DIMENSIONS:
                totals[source][dim] += source_scores[dim]
        wins[row["winner_source"]] += 1

    summary: dict[str, Any] = {
        "num_examples": len(rows),
        "wins": dict(wins),
        "average_scores": {},
    }
    for source in ("base", "lora"):
        summary["average_scores"][source] = {
            dim: round(totals[source][dim] / len(rows), 3) for dim in RUBRIC_DIMENSIONS
        }
    return summary


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running the Gemini judge.")

    rows = load_rows(args.input_path)
    client = genai.Client(api_key=args.api_key)

    judged_rows: list[dict[str, Any]] = []
    for row in rows:
        response = client.models.generate_content(
            model=args.judge_model,
            contents=build_judge_prompt(row),
            config={
                "response_mime_type": "application/json",
                "response_json_schema": RUBRIC_SCHEMA,
            },
        )
        payload = json.loads(response.text)
        scores = validate_scores(payload["scores"])
        reported_winner = normalize_winner(str(payload["winner"]))
        score_winner = winner_from_scores(scores)

        scores_by_source = {
            row["response_a_source"]: scores["A"],
            row["response_b_source"]: scores["B"],
        }
        judged_rows.append(
            {
                **row,
                "judge_model": args.judge_model,
                "scores": scores,
                "scores_by_source": scores_by_source,
                "reported_winner": reported_winner,
                "winner": score_winner,
                "winner_source": map_winner_to_source(row, score_winner),
                "reason": payload["reason"].strip(),
            }
        )

    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)
    write_jsonl(output_path, judged_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summarize(judged_rows), indent=2), encoding="utf-8")

    print(f"Saved judged rows to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
