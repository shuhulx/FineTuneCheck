"""Observable inference backends with faithful chat and batching behavior."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, cast

from finetunecheck.models import InferenceResult, ModelSpec, ModelType
from finetunecheck.utils.device import detect_device


class InferenceBackend(ABC):
    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        probe_name: str = "",
        **generation_settings: Any,
    ) -> list[InferenceResult]: ...

    @abstractmethod
    def get_logprobs(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def cleanup(self) -> None: ...

    @property
    @abstractmethod
    def model_path(self) -> str: ...

    @property
    @abstractmethod
    def backend_name(self) -> str: ...


class TransformersBackend(InferenceBackend):
    """Hugging Face generation with chat templates and correct left padding."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        path: str,
        *,
        adapter_base_model: str | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._path = path
        self.adapter_relationship = (
            {"base_model_name_or_path": adapter_base_model} if adapter_base_model else {}
        )
        self._model.eval()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

    @property
    def model_path(self) -> str:
        return self._path

    @property
    def backend_name(self) -> str:
        return "transformers"

    def _format_prompts(self, prompts: list[str]) -> list[str]:
        if not getattr(self._tokenizer, "chat_template", None):
            return prompts
        return [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

    def _context_limit(self) -> int:
        candidates = [
            getattr(getattr(self._model, "config", None), "max_position_embeddings", None),
            getattr(self._tokenizer, "model_max_length", None),
        ]
        sane = [
            int(value) for value in candidates if isinstance(value, int) and 0 < value < 10_000_000
        ]
        return min(sane) if sane else 2048

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        probe_name: str = "",
        **generation_settings: Any,
    ) -> list[InferenceResult]:
        if not prompts:
            return []
        import torch

        formatted = self._format_prompts(prompts)
        encoded = self._tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_len = int(encoded["input_ids"].shape[1])
        context_limit = self._context_limit()
        if input_len + max_tokens > context_limit:
            raise ValueError(
                f"Prompt batch uses {input_len} tokens and requests {max_tokens} new tokens, "
                f"exceeding context limit {context_limit}"
            )
        if hasattr(encoded, "to"):
            encoded = encoded.to(self._device)
        else:
            encoded = {key: value.to(self._device) for key, value in encoded.items()}

        allowed_settings = {
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
        }
        unknown = set(generation_settings) - allowed_settings
        if unknown:
            raise ValueError(f"Unsupported generation settings: {sorted(unknown)}")
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "pad_token_id": self._tokenizer.pad_token_id,
            **generation_settings,
        }
        started = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(**encoded, **generate_kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if len(output_ids) != len(prompts):
            raise ValueError(
                f"Transformers generated {len(output_ids)} sequences for {len(prompts)} prompts"
            )
        per_sample_ms = elapsed_ms / len(prompts)
        return [
            InferenceResult(
                model_path=self._path,
                probe_name=probe_name,
                sample_id=str(index),
                output=self._tokenizer.decode(
                    token_ids[input_len:], skip_special_tokens=True
                ).strip(),
                latency_ms=per_sample_ms,
                backend=self.backend_name,
            )
            for index, token_ids in enumerate(output_ids)
        ]

    def get_logprobs(self, texts: list[str]) -> list[list[float]]:
        import torch

        all_logprobs: list[list[float]] = []
        for text in self._format_prompts(texts):
            encoded = self._tokenizer(text, return_tensors="pt", truncation=False)
            length = int(encoded["input_ids"].shape[1])
            if length > self._context_limit():
                raise ValueError(
                    f"Input uses {length} tokens, exceeding context limit {self._context_limit()}"
                )
            encoded = encoded.to(self._device)
            with torch.no_grad():
                outputs = self._model(**encoded)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            token_ids = encoded["input_ids"][0]
            all_logprobs.append(
                [
                    log_probs[0, position - 1, token_ids[position]].item()
                    for position in range(1, len(token_ids))
                ]
            )
        return all_logprobs

    def cleanup(self) -> None:
        import torch

        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLLMBackend(InferenceBackend):
    def __init__(self, model_path: str, **kwargs: Any) -> None:
        try:
            from vllm import LLM, SamplingParams
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "vLLM is not installed. Install with: pip install finetunecheck[vllm]"
            ) from exc
        self._path = model_path
        self._llm = LLM(model=model_path, **kwargs)
        self._sampling_params = SamplingParams

    @property
    def model_path(self) -> str:
        return self._path

    @property
    def backend_name(self) -> str:
        return "vllm"

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        probe_name: str = "",
        **generation_settings: Any,
    ) -> list[InferenceResult]:
        if not prompts:
            return []
        params = self._sampling_params(
            max_tokens=max_tokens,
            temperature=generation_settings.pop("temperature", 0.0),
            **generation_settings,
        )
        started = time.perf_counter()
        outputs = self._llm.generate(prompts, params)
        if len(outputs) != len(prompts):
            raise ValueError(f"vLLM generated {len(outputs)} results for {len(prompts)} prompts")
        per_sample_ms = (time.perf_counter() - started) * 1000 / len(prompts)
        return [
            InferenceResult(
                model_path=self._path,
                probe_name=probe_name,
                sample_id=str(index),
                output=(output.outputs[0].text if output.outputs else "").strip(),
                latency_ms=per_sample_ms,
                backend=self.backend_name,
            )
            for index, output in enumerate(outputs)
        ]

    def get_logprobs(self, texts: list[str]) -> list[list[float]]:
        params = self._sampling_params(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        result: list[list[float]] = []
        for text in texts:
            outputs = self._llm.generate([text], params)
            prompt_logprobs = outputs[0].prompt_logprobs or []
            result.append(
                [next(iter(entry.values())).logprob for entry in prompt_logprobs if entry]
            )
        return result

    def cleanup(self) -> None:
        del self._llm


class LlamaCppBackend(InferenceBackend):
    def __init__(self, model_path: str, **kwargs: Any) -> None:
        try:
            from llama_cpp import Llama
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "llama-cpp-python is not installed. Install finetunecheck[gguf]"
            ) from exc
        self._path = model_path
        self._context_size = int(kwargs.pop("n_ctx", 2048))
        self._llm = Llama(
            model_path=model_path,
            n_ctx=self._context_size,
            verbose=False,
            **kwargs,
        )

    @property
    def model_path(self) -> str:
        return self._path

    @property
    def backend_name(self) -> str:
        return "llama_cpp"

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        probe_name: str = "",
        **generation_settings: Any,
    ) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        for index, prompt in enumerate(prompts):
            if len(self._llm.tokenize(prompt.encode())) + max_tokens > self._context_size:
                raise ValueError("Prompt plus requested output exceeds llama.cpp context")
            started = time.perf_counter()
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                echo=False,
                **generation_settings,
            )
            text = output["choices"][0]["text"] if output["choices"] else ""
            results.append(
                InferenceResult(
                    model_path=self._path,
                    probe_name=probe_name,
                    sample_id=str(index),
                    output=text.strip(),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    backend=self.backend_name,
                )
            )
        return results

    def get_logprobs(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            output = self._llm(text, max_tokens=1, echo=True, logprobs=1)
            token_logprobs = (
                output.get("choices", [{}])[0].get("logprobs", {}).get("token_logprobs", [])
            )
            result.append([value for value in token_logprobs if value is not None])
        return result

    def cleanup(self) -> None:
        del self._llm


def _dtype_for_device(resolved_device: str):
    import torch

    if resolved_device == "cpu":
        return torch.float32
    if resolved_device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def create_backend(
    spec: ModelSpec,
    device: str = "auto",
    *,
    preference: str = "auto",
) -> InferenceBackend:
    """Create an explicit/observable backend without swallowing runtime failures."""
    if spec.model_type == ModelType.GGUF:
        if preference not in {"auto", "llama_cpp"}:
            raise ValueError("GGUF models require the llama_cpp backend")
        return LlamaCppBackend(spec.path)
    if preference == "llama_cpp":
        raise ValueError("llama_cpp requires a GGUF model")
    if preference in {"auto", "vllm"} and spec.model_type == ModelType.HF:
        try:
            return VLLMBackend(
                spec.path,
                **({"revision": spec.revision} if spec.revision else {}),
            )
        except ImportError:
            if preference == "vllm":
                raise
    if preference not in {"auto", "transformers", "vllm"}:
        raise ValueError(f"Unknown inference backend: {preference}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = detect_device(device)
    use_device_map_auto = device == "auto" and resolved_device != "cpu"
    dtype = _dtype_for_device(resolved_device)
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if use_device_map_auto else None,
    }

    if spec.model_type == ModelType.LORA:
        if not spec.base_model:
            from peft import PeftConfig

            peft_config = PeftConfig.from_pretrained(spec.path)
            spec = spec.model_copy(update={"base_model": peft_config.base_model_name_or_path})
        if not spec.base_model:
            raise ValueError("PEFT adapter does not declare base_model_name_or_path")
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(spec.base_model, **load_kwargs)
        adapter_kwargs = {"revision": spec.revision} if spec.revision else {}
        model: Any = (
            cast(Any, PeftModel)
            .from_pretrained(base, spec.path, **adapter_kwargs)
            .merge_and_unload()
        )
        tokenizer_source = spec.base_model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            spec.path,
            **load_kwargs,
            **({"revision": spec.revision} if spec.revision else {}),
        )
        tokenizer_source = spec.path
    if not use_device_map_auto:
        model = model.to(resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        **(
            {"revision": spec.revision} if spec.revision and spec.model_type == ModelType.HF else {}
        ),
    )
    input_device = str(getattr(model, "device", resolved_device))
    cache_path = f"{spec.path}@{spec.revision}" if spec.revision else spec.path
    return TransformersBackend(
        model,
        tokenizer,
        input_device,
        cache_path,
        adapter_base_model=(spec.base_model if spec.model_type == ModelType.LORA else None),
    )
