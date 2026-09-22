#!/usr/bin/env python3
"""Run SGLang offline batch inference from a CSV column of prompts.

This script is intentionally small and follows the same high-level pattern as
SGLang's `offline_batch_inference.py` example:

1. Build an `sgl.Engine(...)` from a command-line model id.
2. Convert prompts into rendered strings.
3. Call `llm.generate(prompts, sampling_params)`.

The main difference is the input format. Instead of hard-coded prompt strings,
this script reads a CSV file with `pandas`, then uses one user-specified column
as the batch of prompts.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


Message = Dict[str, str]

def load_prompts_from_csv(path: Path, prompt_column: str) -> List[str]:
    """Read one column of prompt strings from a CSV file with pandas."""

    import pandas as pd

    frame = pd.read_csv(path)

    if prompt_column not in frame.columns:
        raise ValueError(
            f"Column '{prompt_column}' was not found in {path}. "
            f"Available columns: {list(frame.columns)}"
        )

    prompts: List[str] = []
    for row_idx, value in enumerate(frame[prompt_column].tolist()):
        if value is None:
            raise ValueError(
                f"Row {row_idx} in column '{prompt_column}' is empty. "
                "Every prompt must be a string."
            )

        if isinstance(value, float) and value != value:
            raise ValueError(
                f"Row {row_idx} in column '{prompt_column}' is NaN. "
                "Every prompt must be a string."
            )

        if not isinstance(value, str):
            value = str(value)

        prompt_text = value.strip()
        if not prompt_text:
            raise ValueError(
                f"Row {row_idx} in column '{prompt_column}' is blank. "
                "Every prompt must contain text."
            )

        prompts.append(prompt_text)

    if not prompts:
        raise ValueError(
            f"Column '{prompt_column}' in {path} did not contain any prompts."
        )

    return prompts


def prompt_text_to_messages(prompt_text: str) -> List[Message]:
    """Wrap one plain-text prompt as a standard chat message list."""

    return [{"role": "user", "content": prompt_text}]


def render_messages_as_text(tokenizer: Any, messages: List[Message]) -> str:
    """Render chat messages into a text prompt for offline generation."""

    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered_messages = []
    for message in messages:
        rendered_messages.append(
            f"{message['role'].upper()}: {message['content']}".strip()
        )
    rendered_messages.append("ASSISTANT:")
    return "\n\n".join(rendered_messages)


def build_sampling_params(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect the generation settings passed to `llm.generate(...)`."""

    sampling_params: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.top_k is not None:
        sampling_params["top_k"] = args.top_k
    return sampling_params


def run_batch_inference(
    model: str,
    prompts: List[str],
    sampling_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run SGLang offline batch inference and return readable results."""

    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)

    message_batches = [prompt_text_to_messages(prompt) for prompt in prompts]
    rendered_prompts = [
        render_messages_as_text(tokenizer, messages) for messages in message_batches
    ]

    llm = sgl.Engine(model_path=model)
    outputs = llm.generate(rendered_prompts, sampling_params)

    results = []
    for prompt_text, messages, rendered_prompt, output in zip(
        prompts,
        message_batches,
        rendered_prompts,
        outputs,
    ):
        generated_text = output["text"] if isinstance(output, dict) else str(output)
        results.append(
            {
                "input_prompt": prompt_text,
                "messages": messages,
                "rendered_prompt": rendered_prompt,
                "generated_text": generated_text,
                "raw_output": output,
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run SGLang offline batch inference from a CSV prompt column."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id or local model path to run with SGLang.",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        required=True,
        help="CSV file containing a column of prompts.",
    )
    parser.add_argument(
        "--prompt-column",
        required=True,
        help="Name of the CSV column to use for prompts.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional JSON file to write results to. Defaults to stdout.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def main() -> None:
    """Parse CLI args, run inference, and write JSON results."""

    parser = build_parser()
    args = parser.parse_args()

    prompts = load_prompts_from_csv(args.csv_file, args.prompt_column)
    sampling_params = build_sampling_params(args)
    results = run_batch_inference(args.model, prompts, sampling_params)

    output_text = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output_file:
        args.output_file.write_text(output_text + "\n", encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
