"""Setuptools compatibility entrypoint for packaging llm-utils."""

from pathlib import Path

from setuptools import find_packages, setup


VERSION = "0.1.0"
README = Path(__file__).with_name("DOCS.md").read_text(encoding="utf-8")


setup(
    name="llm-utils",
    version=VERSION,
    description="Utilities for working with local and server-backed LLM pipelines.",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    packages=find_packages(include=["llm_utils", "llm_utils.*"]),
    install_requires=[
        "requests>=2.20.0",
    ],
    extras_require={
        "anthropic": [
            "anthropic>=0.30.0",
        ],
        "dev": [
            "pytest>=6.2.0",
        ],
        "minimax": [
            "openai>=1.0.0",
        ],
        "offline-batch": [
            "pandas>=2.0.0",
            "sglang>=0.5.0",
            "transformers>=4.40.0",
        ],
        "sglang": [
            "sglang>=0.5.0",
        ],
        "transformers": [
            "torch>=2.0.0",
            "transformers>=4.40.0",
        ],
        "vllm": [
            "vllm>=0.5.0",
        ],
        "all": [
            "anthropic>=0.30.0",
            "openai>=1.0.0",
            "pandas>=2.0.0",
            "sglang>=0.5.0",
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "vllm>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-utils=llm_utils.cli:main",
            "llm-utils-deploy-sglang=llm_utils.deploy_sglang:main",
            (
                "llm-utils-sglang-offline-batch="
                "llm_utils.sglang_offline_batch_inference:main"
            ),
        ],
    },
)
