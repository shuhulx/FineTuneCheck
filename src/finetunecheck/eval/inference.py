"""Inference backend abstraction — pluggable generation for HF, vLLM, and llama.cpp."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import torch

from finetunecheck.models import InferenceResult, ModelSpec, ModelType
from finetunecheck.utils.device import detect_device


class InferenceBackend(ABC):
    @abstractmethod
    def generate_batch(
        self, prompts: list[str], max_tokens: int = 512, probe_name: str = ""
    ) -> list[InferenceResult]: ...

    @abstractmethod
    def get_logprobs(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def cleanup(self) -> None: ...

    @property
    @abstractmethod
    def model_path(self) -> str: ...


class TransformersBackend(InferenceBackend):
    """Uses HuggingFace transformers for generation."""

    def __init__(self, model, tokenizer, device: str, path: str) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._path = path
        self._model.eval()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def model_path(self) -> str:
        return self._path

    def generate_batch(
        self, prompts: list[str], max_tokens: int = 512, probe_name: str = ""
    ) -> list[InferenceResult]:
        results = []
        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self._device)

        input_len = encoded["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_sample_ms = elapsed_ms / len(prompts)

        for i, ids in enumerate(output_ids):
            generated_ids = ids[input_len:]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(
                InferenceResult(
                    model_path=self._path,
                    probe_name=probe_name,
                    sample_id=str(i),
                    output=text.strip(),
                    latency_ms=per_sample_ms,
                )
            )
        return results

    def get_logprobs(self, texts: list[str]) -> list[list[float]]:
        all_logprobs = []
        for text in texts:
            encoded = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=2048
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**encoded)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            token_ids = encoded["input_ids"][0]
            token_logprobs = []
            for pos in range(1, len(token_ids)):
                lp = log_probs[0, pos - 1, token_ids[pos]].item()
                token_logprobs.append(lp)
            all_logprobs.append(token_logprobs)
        return all_logprobs

    def cleanup(self) -> None:
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLLMBackend(InferenceBackend):
    """Uses vLLM for fast batch inference."""

    def __init__(self, model_path: str, **kwargs) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install with: pip install finetunecheck[vllm]"
            )
        self._path = model_path
        self._llm = LLM(model=model_path, **kwargs)
        self._SamplingParams = SamplingParams

    @property
    def model_path(self) -> str:
        return self._path

    def generate_batch(
        self, prompts: list[str], max_tokens: int = 512, probe_name: str = ""
    ) -> list[InferenceResult]:
        params = self._SamplingParams(max_tokens=max_tokens, temperature=0.0)
        t0 = time.perf_counter()
        outputs = self._llm.generate(prompts, params)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_sample_ms = elapsed_ms / len(prompts) if prompts else 0

        results = []
        for i, out in enumerate(outputs):
            text = out.outputs[0].text if out.outputs else ""
            results.append(
                InferenceResult(
                    model_path=self._path,
                    probe_name=probe_name,
                    sample_id=str(i),
                    output=text.strip(),
                    latency_ms=per_sample_ms,
                )
            )
        return results

    def get_logprobs(self, texts: list[str]) -> list[list[float]]:
        params = self._SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        all_logprobs = []
        for text in texts:
            outputs = self._llm.generate([text], params)
            prompt_lps = outputs[0].prompt_logprobs or []
            token_lps = []
            for lp_dict in prompt_lps:
                if lp_dict is not None:
                    vals = list(lp_dict.values())
                    token_lps.append(vals[0].logprob if vals else 0.0)
            all_logprobs.append(token_lps)
        return all_logprobs

    def cleanup(self) -> None:
        del self._llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LlamaCppBackend(InferenceBackend):
    """Uses llama-cpp-python for GGUF models."""

    def __init__(self, model_path: str, **kwargs) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. Install with: pip install finetunecheck[gguf]"
            )
        self._path = model_path
        n_ctx = kwargs.pop("n_ctx", 2048)
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False, **kwargs)

    @property
    def model_path(self) -> str:
        return self._path

    def generate_batch(
        self, prompts: list[str], max_tokens: int = 512, probe_name: str = ""
    ) -> list[InferenceResult]:
        results = []
        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            out = self._llm(prompt, max_tokens=max_tokens, echo=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            text = out["choices"][0]["text"] if out["choices"] else ""
            results.append(
                InferenceResult(
                    model_path=self._path,
                    probe_name=probe_name,
                    sample_id=str(i),
                    output=text.strip(),
                    latency_ms=elapsed_ms,
                )
            )
        return results

    def get_logprobs(self, texts: list[str]) -> list[list[float]]:
        all_logprobs = []
        for text in texts:
            out = self._llm(text, max_tokens=1, echo=True, logprobs=1)
            lps = []
            for tok_info in (
                out.get("choices", [{}])[0].get("logprobs", {}).get("token_logprobs", [])
            ):
                if tok_info is not None:
                    lps.append(tok_info)
            all_logprobs.append(lps)
        return all_logprobs

    def cleanup(self) -> None:
        del self._llm


def create_backend(spec: ModelSpec, device: str = "auto") -> InferenceBackend:
    """Factory: pick the best available backend for the given model spec.

    Priority for HF/LoRA: vLLM > transformers.
    GGUF always uses llama.cpp.
    """
    resolved_device = detect_device(device)

    if spec.model_type == ModelType.GGUF:
        return LlamaCppBackend(spec.path)

    # Try vLLM first for HF models (LoRA support varies)
    if spec.model_type == ModelType.HF:
        try:
            return VLLMBackend(spec.path)
        except (ImportError, Exception):
            pass

    # Transformers fallback
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if spec.model_type == ModelType.LORA:
        if not spec.base_model:
            raise ValueError(
                "LoRA adapter detected but base_model not set in ModelSpec. "
                "Ensure adapter_config.json contains base_model_name_or_path."
            )
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            spec.base_model,
            torch_dtype=torch.float16,
            device_map=resolved_device if resolved_device == "auto" else None,
        )
        model = PeftModel.from_pretrained(base, spec.path)
        model = model.merge_and_unload()
        if resolved_device not in ("auto",):
            model = model.to(resolved_device)
        tokenizer = AutoTokenizer.from_pretrained(spec.base_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            spec.path,
            torch_dtype=torch.float16,
            device_map=resolved_device if resolved_device == "auto" else None,
        )
        if resolved_device not in ("auto",):
            model = model.to(resolved_device)
        tokenizer = AutoTokenizer.from_pretrained(spec.path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actual_device = resolved_device if resolved_device != "auto" else "cpu"
    return TransformersBackend(model, tokenizer, actual_device, spec.path)
