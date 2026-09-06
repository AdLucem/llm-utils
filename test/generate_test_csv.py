import argparse
import csv
import json
import random
from pathlib import Path


SYSTEM_PROMPTS = (
    "You are a brisk recipe muse.",
    "You are a tiny sci-fi narrator.",
    "You are a calm code reviewer.",
    "You are a witty travel guide.",
    "You are a poetic weather bot.",
    "You are a sharp museum docent.",
    "You are a playful math coach.",
    "You are a concise debate judge.",
)

USER_PROMPTS = (
    "Name a sandwich for astronauts.",
    "Explain rain to a dragon.",
    "Pitch a library on the moon.",
    "Write a motto for quiet robots.",
    "Describe a cactus at midnight.",
    "Invent a festival for socks.",
    "Summarize thunder in six words.",
    "Give tea advice to pirates.",
)


def build_rows(n: int) -> list[dict[str, object]]:
    total_rows = 2 * n
    labels = [0] * n + [1] * n
    random.shuffle(labels)
    rng = random.Random()

    rows = []
    for row_id, label in enumerate(labels):
        prompt = [
            {
                "role": "system",
                "content": rng.choice(SYSTEM_PROMPTS)
            },
            {
                "role": "user",
                "content": rng.choice(USER_PROMPTS)
            }
        ]
        rows.append(
            {
                "id": row_id,
                "label": label,
                "prompt": json.dumps(prompt, separators=(",", ":")),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="Half the number of rows to generate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_test.csv",
        help="Output CSV path. Defaults to this directory.",
    )
    args = parser.parse_args()

    if args.n < 0:
        raise ValueError("n must be non-negative")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("id", "label", "prompt"))
        writer.writeheader()
        writer.writerows(build_rows(args.n))


if __name__ == "__main__":
    main()
