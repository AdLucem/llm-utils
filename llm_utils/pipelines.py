import logging
import pathlib
import re
from collections import defaultdict
from typing import List, Optional, Union

from ._compat import Literal, dataclass
from .llm_configs import args_to_request_config
from .request_minimax import minimax_chat_completion, minimax_chat_completion_batch
from .request_sglang import (
    configure_logging,
    sglang_chat_completion,
    sglang_chat_completion_batch,
)

PIPELINE_TYPES = Literal["sglang", "transformers", "minimax", "mock"]
LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


@dataclass
class PipelineConfig:

    # Pipeline arguments
    model: str
    pipeline_type: PIPELINE_TYPES = "sglang"
    log_level: LOG_LEVELS = "INFO"

    # Inputs
    init_prompts: Optional[pathlib.Path] = None

    # Model hyperparameters
    temperature: float = 0.7
    max_new_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 50

    # SGLang pipeline arguments
    host: Optional[str] = "127.0.0.1"
    port: Optional[int] = 30000
    timeout: Optional[int] = 180  # seconds

    # Transformers pipeline arguments
    device: Optional[Literal["auto", "cpu", "cuda", "mps"]] = "auto"
    dtype: Optional[Literal["auto", "float16", "bfloat16", "float32"]] = "auto"


class LLMPipeline:

    def __init__(self, cfg: PipelineConfig):

        self.model_name = cfg.model

    def parse_inputs(self, inputs):
        """
        1. If `inputs` is a string, parse it into a [{"role": "user", "content": <string>}] singleton list
        2. If inputs is a list of strings, parse them into a [{"role": "user", "content": <string>}] list
        3. If incoming inputs is a list of dicts, return as-is.
        """

        parallel = False

        if isinstance(inputs, str):
            logging.debug("Input to pipeline: single string")
            messages = [{"role": "user", "content": inputs}]

        elif isinstance(inputs, list) and inputs:
            first = inputs[0]

            if isinstance(first, str):
                logging.debug("Input to pipeline: list of strings")
                messages = [{"role": "user", "content": inp} for inp in inputs]
            elif isinstance(first, dict):
                logging.debug("Input to pipeline: List of dicts")
                messages = inputs
            elif isinstance(first, list) and first and isinstance(first[0], dict):
                logging.debug("Input to pipeline: List of lists")
                messages = inputs
                parallel = True
            else:
                logging.error("Invalid input type: %s", inputs)
                return ""
        else:
            logging.error("Invalid input type: %s", inputs)
            return ""

        return messages, parallel

    def generate(self, inputs):

        logging.error(
            "It seems that you have accidentally used the base AgentPipeline class, "
            "which does not have a `generate` implementation."
        )


class TransformersPipeline(LLMPipeline):

    def __init__(self, cfg: PipelineConfig):

        super().__init__(cfg)

        self.cfg = cfg
        self.temperature = cfg.temperature
        self.max_new_tokens = cfg.max_new_tokens
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k
        self.device = cfg.device
        self.dtype = cfg.dtype
        configure_logging(self.cfg.log_level)

        self.tokenizer, self.model = self._init_transformers(cfg)

    def _resolve_torch_dtype(self):
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype)

    def _init_transformers(self, cfg: PipelineConfig):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required to use TransformersPipeline."
            ) from exc

        model_kwargs = {}
        torch_dtype = self._resolve_torch_dtype()
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        if cfg.device == "auto":
            model_kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        model = AutoModelForCausalLM.from_pretrained(cfg.model, **model_kwargs)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if cfg.device != "auto":
            model = model.to(cfg.device)

        model.eval()
        return tokenizer, model

    def _messages_to_prompt(self, messages):
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        rendered_messages = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            if content:
                rendered_messages.append(f"{role}: {content}")
        rendered_messages.append("ASSISTANT:")
        return "\n\n".join(rendered_messages)

    def _prepare_inputs(self, prompts):
        import torch

        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if self.device != "auto":
            tokenized = {
                key: value.to(self.device)
                for key, value in tokenized.items()
            }
        elif hasattr(self.model, "device") and self.model.device.type != "cpu":
            tokenized = {
                key: value.to(self.model.device)
                for key, value in tokenized.items()
            }

        return tokenized

    def _decode_responses(self, generated_ids, attention_mask):
        responses = []
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        for row_idx, prompt_length in enumerate(prompt_lengths):
            generated_tokens = generated_ids[row_idx][prompt_length:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            responses.append({"role": "assistant", "content": text})

        return responses

    def _generate_from_message_batches(self, message_batches):
        import torch

        prompts = [self._messages_to_prompt(messages) for messages in message_batches]
        model_inputs = self._prepare_inputs(prompts)

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.top_k is not None and self.top_k > 0:
            generation_kwargs["top_k"] = self.top_k

        if self.temperature and self.temperature > 0:
            generation_kwargs["do_sample"] = True
        else:
            generation_kwargs["do_sample"] = False
            generation_kwargs["temperature"] = None
            generation_kwargs["top_p"] = None
            generation_kwargs.pop("top_k", None)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                **generation_kwargs,
            )

        return self._decode_responses(generated_ids, model_inputs["attention_mask"])

    def generate(self, inputs) -> Union[dict, List[dict]]:
        messages, parallel = super().parse_inputs(inputs)

        if parallel:
            response = self._generate_from_message_batches(messages)
            debug_msg = "\n" + ("=" * 60) + "\n"
            debug_msg += "Transformers query responses:\n"
            for r in response:
                debug_msg += f"{r}\n"
                debug_msg += ("-" * 40) + "\n"
            debug_msg += ("=" * 60) + "\n"
        else:
            response = self._generate_from_message_batches([messages])[0]
            debug_msg = "\n" + ("=" * 60) + "\n"
            debug_msg += f"Transformers query response: {response}\n"
            debug_msg += ("=" * 60) + "\n"

        logging.debug(debug_msg)
        return response


class SGLangPipeline(LLMPipeline):

    def __init__(self, cfg: PipelineConfig):

        super().__init__(cfg)

        self.cfg = cfg
        configure_logging(self.cfg.log_level)

    def generate(self, inputs) -> Union[str, List[str]]:

        messages, parallel = super().parse_inputs(inputs)

        if parallel:
            response = sglang_chat_completion_batch(cfg=self.cfg, requests_messages=messages)

            debug_msg = "\n" + ("=" * 60) + "\n"
            debug_msg += "SGLANG Query responses:\n"
            for r in response:
                debug_msg += f"{r}\n"
                debug_msg += ("-" * 40) + "\n"
            debug_msg += ("=" * 60) + "\n"
            logging.debug(debug_msg)

        else:
            logging.debug("SGLANG Query messages: %s", messages)
            response = sglang_chat_completion(cfg=self.cfg, messages=messages)
            debug_msg = "\n" + ("=" * 60) + "\n"
            debug_msg += f"SGLANG Query response: {response}\n"
            debug_msg += ("=" * 60) + "\n"
            logging.debug(debug_msg)

        logging.debug(debug_msg)
        return response


class MinimaxPipeline(LLMPipeline):

    def __init__(self, cfg: PipelineConfig):

        super().__init__(cfg)

        self.cfg = cfg
        configure_logging(self.cfg.log_level)

    def generate(self, inputs) -> Union[str, List[str]]:

        messages, parallel = super().parse_inputs(inputs)

        if parallel:
            response = minimax_chat_completion_batch(
                cfg=self.cfg,
                requests_messages=messages,
            )

            debug_msg = "\n" + ("=" * 60) + "\n"
            debug_msg += "MiniMax Query responses:\n"
            for r in response:
                debug_msg += f"{r}\n"
                debug_msg += ("-" * 40) + "\n"
            debug_msg += ("=" * 60) + "\n"
            logging.debug(debug_msg)

        else:
            logging.debug("MiniMax Query messages: %s", messages)
            response = minimax_chat_completion(cfg=self.cfg, messages=messages)
            debug_msg = "\n" + ("=" * 60) + "\n"
            debug_msg += f"MiniMax Query response: {response}\n"
            debug_msg += ("=" * 60) + "\n"
            logging.debug(debug_msg)

        logging.debug(debug_msg)
        return response


class MockPipeline(LLMPipeline):
    """Mock pipeline for testing without a GPU.

    The implementation avoids heavyweight model dependencies by building a
    cached trigram language model from public-domain text. A small built-in
    corpus remains as a resilience fallback when downloads are unavailable.
    """

    FALLBACK_CORPUS_PATH = (
        pathlib.Path(__file__).resolve().parent / ".data/mock_pipeline_fallback_corpus.txt"
    )
    FALLBACK_CORPUS_TEXT = (
        "This is a lightweight fallback corpus for the mock pipeline. "
        "It keeps the import path stable and provides a minimal local text source "
        "when no cached or downloaded corpus is available."
    )
    CORPUS_URLS = (
        "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    )
    CORPUS_CACHE_DIR = pathlib.Path(__file__).resolve().parent / ".mock_pipeline_cache"
    CORPUS_CACHE_FILE = CORPUS_CACHE_DIR / "trigram_corpus.txt"
    MAX_CORPUS_TOKENS = 250_000
    MIN_DOWNLOADED_CHARS = 1_500_000

    def __init__(self, cfg: PipelineConfig):
        """Initialize the cached trigram generator used by the mock pipeline."""

        super().__init__(cfg)

        self.cfg = cfg
        configure_logging(self.cfg.log_level)
        import random

        # Keep generations bounded so the mock pipeline stays fast and stable.
        self.max_generation_tokens = max(8, min(int(cfg.max_new_tokens), 96))
        self._rng = random.Random(0)
        self.model = {}
        self.prefixes = ()
        self.sentence_starts = ()
        self.prefix_by_first = defaultdict(list)
        self.prefix_by_second = defaultdict(list)
        self.fallback_corpus = self._load_fallback_corpus()
        self._init_ngram_generator()

    def _load_fallback_corpus(self):
        if self.FALLBACK_CORPUS_PATH.exists():
            return self.FALLBACK_CORPUS_PATH.read_text(encoding="utf-8", errors="ignore")

        logging.warning(
            "Mock pipeline fallback corpus not found at %s.",
            self.FALLBACK_CORPUS_PATH,
        )
        return self.FALLBACK_CORPUS_TEXT

    def _init_ngram_generator(self):
        """Download and cache a public corpus, then build trigram transitions."""

        import urllib.request

        def strip_gutenberg_boilerplate(text):
            start_markers = (
                "*** START OF THE PROJECT GUTENBERG EBOOK",
                "*** START OF THIS PROJECT GUTENBERG EBOOK",
            )
            end_markers = (
                "*** END OF THE PROJECT GUTENBERG EBOOK",
                "*** END OF THIS PROJECT GUTENBERG EBOOK",
            )

            start_index = 0
            for marker in start_markers:
                found = text.find(marker)
                if found != -1:
                    start_index = text.find("\n", found)
                    start_index = 0 if start_index == -1 else start_index
                    break

            end_index = len(text)
            for marker in end_markers:
                found = text.find(marker)
                if found != -1:
                    end_index = found
                    break

            return text[start_index:end_index]

        def tokenize(text):
            return re.findall(r"[A-Za-z0-9']+|[.!?,;:]", text)

        corpus_text = ""

        try:
            if self.CORPUS_CACHE_FILE.exists():
                corpus_text = self.CORPUS_CACHE_FILE.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logging.debug("Unable to read cached mock corpus: %s", exc)

        if not corpus_text:
            downloaded_parts = []
            for url in self.CORPUS_URLS:
                try:
                    request = urllib.request.Request(
                        url,
                        headers={"User-Agent": "mock-pipeline/1.0"},
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        downloaded_parts.append(
                            strip_gutenberg_boilerplate(
                                response.read().decode("utf-8", errors="ignore")
                            )
                        )
                    if sum(len(part) for part in downloaded_parts) >= self.MIN_DOWNLOADED_CHARS:
                        break
                except Exception as exc:
                    logging.debug("Unable to download mock corpus from %s: %s", url, exc)

            if downloaded_parts:
                corpus_text = "\n".join(downloaded_parts)
                try:
                    self.CORPUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    self.CORPUS_CACHE_FILE.write_text(corpus_text, encoding="utf-8")
                except OSError as exc:
                    logging.debug("Unable to cache downloaded mock corpus: %s", exc)

        if not corpus_text:
            logging.warning("Falling back to built-in mock corpus after download failure.")
            corpus_text = self.fallback_corpus

        tokens = tokenize(corpus_text)
        if len(tokens) > self.MAX_CORPUS_TOKENS:
            tokens = tokens[:self.MAX_CORPUS_TOKENS]

        transitions = defaultdict(lambda: defaultdict(int))
        sentence_starts = defaultdict(int)
        sentence_boundary = {".", "!", "?"}

        for i in range(len(tokens) - 2):
            prefix = (tokens[i], tokens[i + 1])
            next_token = tokens[i + 2]
            transitions[prefix][next_token] += 1

            if i == 0 or tokens[i - 1] in sentence_boundary:
                sentence_starts[prefix] += 1

        self.model = {}
        for prefix, next_counts in transitions.items():
            next_tokens = tuple(next_counts.keys())
            weights = tuple(next_counts.values())
            self.model[prefix] = (next_tokens, weights)
            self.prefix_by_first[prefix[0]].append(prefix)
            self.prefix_by_second[prefix[1]].append(prefix)

        self.prefixes = tuple(self.model.keys())
        if sentence_starts:
            ranked_starts = sorted(sentence_starts.items(), key=lambda item: item[1], reverse=True)
            self.sentence_starts = tuple(prefix for prefix, _ in ranked_starts)
        else:
            self.sentence_starts = self.prefixes

    def predict_next_word(self, w1, w2):
        next_word_options = self.model.get((w1, w2))
        if not next_word_options:
            return "No prediction available"

        tokens, weights = next_word_options
        return self._rng.choices(tokens, weights=weights, k=1)[0]

    def _pick_prefix(self, prompt_tokens):
        for idx in range(len(prompt_tokens) - 1, 0, -1):
            prefix = (prompt_tokens[idx - 1], prompt_tokens[idx])
            if prefix in self.model:
                return prefix, True

        if prompt_tokens:
            last_token = prompt_tokens[-1]
            candidates = (
                self.prefix_by_first.get(last_token)
                or self.prefix_by_second.get(last_token)
                or self.sentence_starts
                or self.prefixes
            )
            return self._rng.choice(candidates), False

        return self._rng.choice(self.sentence_starts or self.prefixes), False

    def _detokenize(self, tokens):
        if not tokens:
            return ""

        no_space_before = {".", ",", "!", "?", ";", ":"}
        pieces = []

        for token in tokens:
            if not pieces:
                pieces.append(token)
            elif token in no_space_before:
                pieces[-1] += token
            else:
                pieces.append(" " + token)

        return "".join(pieces)

    def generate_sentence(self, input_sentence, max_new_tokens=100):

        input_tokens = re.findall(r"[A-Za-z0-9']+|[.!?,;:]", input_sentence)
        if not self.model:
            return input_sentence

        prefix, used_prompt_context = self._pick_prefix(input_tokens)
        output_tokens = [] if used_prompt_context else [prefix[0], prefix[1]]
        sentence_boundary = {".", "!", "?"}

        for _ in range(max_new_tokens):
            w3 = self.predict_next_word(prefix[0], prefix[1])
            if w3 == "No prediction available":
                prefix = self._rng.choice(self.sentence_starts or self.prefixes)
                continue

            if not output_tokens and w3 in {".", ",", "!", "?", ";", ":"}:
                prefix = (prefix[1], w3)
                continue

            output_tokens.append(w3)
            prefix = (prefix[1], w3)

            if len(output_tokens) >= 12 and w3 in sentence_boundary:
                break

        return self._detokenize(output_tokens).strip()

    def generate(self, inputs) -> Union[str, List[str]]:

        def response_from_message_list(message_list):
            non_system_messages = [
                msg.get("content", "")
                for msg in message_list
                if msg.get("role") != "system" and msg.get("content")
            ]
            relevant_messages = non_system_messages or [
                msg.get("content", "") for msg in message_list if msg.get("content")
            ]
            prompt = "\n".join(relevant_messages[-3:]).strip()
            return self.generate_sentence(prompt, max_new_tokens=self.max_generation_tokens)

        messages, parallel = super().parse_inputs(inputs)

        if parallel:
            response = [
                {"role": "assistant", "content": response_from_message_list(message_list)}
                for message_list in messages
            ]

            debug_msg = "\n" + ("=" * 60) + "\n"
            debug_msg += "Mock Query responses:\n"
            for r in response:
                debug_msg += f"{r}\n"
                debug_msg += ("-" * 40) + "\n"
            debug_msg += ("=" * 60) + "\n"

        elif isinstance(inputs, list) and inputs and isinstance(inputs[0], str):
            response = [
                self.generate_sentence(msg["content"], max_new_tokens=self.max_generation_tokens)
                for msg in messages
            ]
            response = [{"role": "assistant", "content": r} for r in response]
            debug_msg = "\n" + ("=" * 60) + "\n"
            debug_msg += f"Mock Query response: {response}\n"
            debug_msg += ("=" * 60) + "\n"

        elif isinstance(inputs, list) and inputs and isinstance(inputs[0], dict):
            response = response_from_message_list(messages)
            response = {"role": "assistant", "content": response}
            debug_msg = "\n" + ("=" * 60) + "\n"
            debug_msg += f"Mock Query response: {response}\n"
            debug_msg += ("=" * 60) + "\n"

        else:
            response = self.generate_sentence(
                messages[0]["content"],
                max_new_tokens=self.max_generation_tokens,
            )
            response = {"role": "assistant", "content": response}
            debug_msg = "\n" + ("=" * 60) + "\n"
            debug_msg += f"Mock Query response: {response}\n"
            debug_msg += ("=" * 60) + "\n"

        logging.debug(debug_msg)

        return response


def pipeline_from_config(cfg: PipelineConfig):

    if cfg.pipeline_type == "sglang":
        llm_pipeline = SGLangPipeline(cfg)
    elif cfg.pipeline_type == "transformers":
        llm_pipeline = TransformersPipeline(cfg)
    elif cfg.pipeline_type == "minimax":
        llm_pipeline = MinimaxPipeline(cfg)
    elif cfg.pipeline_type == "mock":
        llm_pipeline = MockPipeline(cfg)
    else:
        raise Exception(f"Incorrect pipeline type {cfg.pipeline_type}")
    return llm_pipeline


def pipeline_config_from_args(args):

    try:
        init_prompts = args.init_prompts
    except AttributeError:
        init_prompts = None

    pipeline_cfg = PipelineConfig(
        model=args.model,
        pipeline_type=args.pipeline_type.lower(),
        log_level=args.log_level,
        init_prompts=init_prompts,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        device=args.device,
        dtype=args.dtype,
    )

    return pipeline_cfg


__all__ = [
    "LLMPipeline",
    "LOG_LEVELS",
    "MinimaxPipeline",
    "MockPipeline",
    "PIPELINE_TYPES",
    "PipelineConfig",
    "SGLangPipeline",
    "TransformersPipeline",
    "args_to_request_config",
    "pipeline_config_from_args",
    "pipeline_from_config",
]
