#!/usr/bin/env python3
"""CLI entrypoint for sending a prompt-file request through SGLangPipeline."""

import argparse
import json
from pathlib import Path

from .pipelines import PipelineConfig, SGLangPipeline


def build_parser():
    """Create the command-line parser for the SGLang pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="Send a system/user prompt file through an SGLang pipeline.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name reported to the SGLang server.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="Path to a JSON or sectioned text file containing system and user prompts.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="SGLang server host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="SGLang server port (default: 30000).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to request (default: 2048).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling value stored in the pipeline config (default: 0.9).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling value stored in the pipeline config (default: 50).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Request timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser


def parse_prompt_file(prompt_file):
    """Load a prompt file with `system` and `user` content."""
    text = prompt_file.read_text(encoding="utf-8")

    if prompt_file.suffix.lower() == ".json":
        data = json.loads(text)
        system_prompt = data.get("system", "").strip()
        user_prompt = data.get("user", "").strip()
    else:
        upper_text = text.upper()
        system_tag = "[SYSTEM]"
        user_tag = "[USER]"

        system_start = upper_text.find(system_tag)
        user_start = upper_text.find(user_tag)

        if system_start == -1 or user_start == -1:
            raise ValueError(
                "Plain-text prompt files must contain both [SYSTEM] and [USER] sections."
            )

        system_content_start = system_start + len(system_tag)
        user_content_start = user_start + len(user_tag)

        if system_start < user_start:
            system_prompt = text[system_content_start:user_start].strip()
            user_prompt = text[user_content_start:].strip()
        else:
            user_prompt = text[user_content_start:system_start].strip()
            system_prompt = text[system_content_start:].strip()

    if not system_prompt:
        raise ValueError("Prompt file is missing system prompt content.")
    if not user_prompt:
        raise ValueError("Prompt file is missing user prompt content.")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def validate_args(args):
    """Validate CLI arguments before building the pipeline."""
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port must be between 1 and 65535.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0.")
    if not args.prompt_file.exists():
        raise FileNotFoundError("Prompt file not found: {}".format(args.prompt_file))


def main():
    """Parse args, load prompts, run SGLangPipeline, and print the response."""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    messages = parse_prompt_file(args.prompt_file)

    cfg = PipelineConfig(
        model=args.model,
        pipeline_type="sglang",
        log_level=args.log_level,
        init_prompts=args.prompt_file,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )

    pipeline = SGLangPipeline(cfg)
    response = pipeline.generate(messages)

    if isinstance(response, dict):
        print(response.get("content", ""))
    else:
        print(response)


if __name__ == "__main__":
    main()
