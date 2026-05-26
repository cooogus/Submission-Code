import json
import csv
from pathlib import Path

BENCHMARKS = [
    {
        "benchmark": "agieval_lsat_ar",
        "root": Path("outputs/agieval/lsat_ar/openrouter"),
    },
    {
        "benchmark": "aime_2025",
        "root": Path("outputs/aime/aime_2025/openrouter"),
    },
    {
        "benchmark": "gpqa_diamond",
        "root": Path("outputs/gpqa/gpqa_diamond/openrouter"),
    },  
    {
        "benchmark": "mmmu_accounting",
        "root": Path("outputs/mmmu/mmmu_accounting/openrouter"),
    },
    {
        "benchmark": "mmmu_clinical_medicine",
        "root": Path("outputs/mmmu/mmmu_clinical_medicine/openrouter"),
    },
    {
        "benchmark": "mmmu_finance_merged",
        "root": Path("outputs/mmmu/mmmu_finance_merged/openrouter"),
    },
]


def extract_assistant_text(messages_for_attempt, msg_type):
    for msg in messages_for_attempt:
        if msg.get("role") == "assistant" and msg.get("type") == msg_type:
            return msg.get("content", "")
    return ""


def safe_get_cost(record, key):
    cost = record.get("cost", {})
    if isinstance(cost, dict):
        return cost.get(key)
    return None


def should_skip_file(json_file):
    name = json_file.name.lower()
    return (
        "leaderboard" in name
        or "heatmap" in name
        or "summary" in name
    )


for bench in BENCHMARKS:

    benchmark_name = bench["benchmark"]
    root = bench["root"]

    OUTPUT_CSV = f"{benchmark_name}_clean_attempts.csv"
    VERIFY_CSV = f"{benchmark_name}_verification.csv"

    attempt_rows = []
    verify_rows = []

    print(f"\nProcessing benchmark: {benchmark_name}")

    if not root.exists():
        print(f"Missing folder: {root}")
        continue

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        for json_file in model_dir.rglob("*.json"):

            if should_skip_file(json_file):
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    record = json.load(f)

            except Exception as e:
                print(f"Skipping {json_file}: {e}")
                continue

            question_id = record.get("idx")
            problem = record.get("problem", "")
            gold_answer = record.get("gold_answer")
            source = record.get("source")
            n = record.get("N")

            answers = record.get("answers", [])
            correct = record.get("correct", [])
            messages = record.get("messages", [])

            has_4_attempts = (
                n == 4
                and len(answers) == 4
                and len(correct) == 4
                and len(messages) == 4
            )

            has_reasoning = False

            max_attempts = min(
                len(messages),
                len(answers),
                len(correct)
            )

            for attempt_id in range(max_attempts):

                attempt_messages = messages[attempt_id]

                reasoning_text = extract_assistant_text(
                    attempt_messages,
                    "cot"
                )

                final_response = extract_assistant_text(
                    attempt_messages,
                    "response"
                )

                if reasoning_text:
                    has_reasoning = True

                attempt_rows.append({
                    "benchmark": benchmark_name,
                    "model": model_name,
                    "question_id": question_id,
                    "attempt_id": attempt_id + 1,
                    "gold_answer": gold_answer,
                    "answer": answers[attempt_id],
                    "correct": correct[attempt_id],
                    "reasoning_text": reasoning_text,
                    "final_response": final_response,
                    "problem": problem,
                    "source": source,
                    "json_path": str(json_file),
                    "total_cost": safe_get_cost(record, "cost"),
                    "input_tokens": safe_get_cost(record, "input_tokens"),
                    "output_tokens": safe_get_cost(record, "output_tokens"),
                    "latency_time": safe_get_cost(record, "time"),
                })

            verify_rows.append({
                "benchmark": benchmark_name,
                "model": model_name,
                "json_path": str(json_file),
                "question_id": question_id,
                "N": n,
                "num_answers": len(answers),
                "num_correct": len(correct),
                "num_messages": len(messages),
                "has_4_attempts": has_4_attempts,
                "has_reasoning": has_reasoning,
            })

    if attempt_rows:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=attempt_rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(attempt_rows)

    if verify_rows:
        with open(VERIFY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=verify_rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(verify_rows)

    print(f"Saved attempts CSV: {OUTPUT_CSV}")
    print(f"Saved verification CSV: {VERIFY_CSV}")
    print(f"Total attempts: {len(attempt_rows)}")
    print(f"Total question records: {len(verify_rows)}")