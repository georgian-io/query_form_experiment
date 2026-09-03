"""LLM clients for query transformation (§5.2).

Two provider paths behind one interface -- Vertex for the Gemini family, Novita's
OpenAI-compatible endpoint for open models. Each carries `lineage` metadata so the §1.2
cross-generator control can group by model family: if a leaderboard reshuffle flips entirely
when the generator changes lineage, we are largely measuring self-preference bias rather than a
property of LLM-written queries.

Every generator reports the exact model version string the provider returns, which goes into the
cached record. Provider aliases move -- `gemini-2.5-pro` resolves to different snapshots over
time -- so the alias alone is not provenance.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import GeneratorConfig, Provider


@dataclass(frozen=True)
class Completion:
    text: str
    model_version: str
    finish_reason: str = ""


class TruncatedCompletion(RuntimeError):
    """A completion that stopped early rather than finishing.

    Raised rather than returned because a truncated rewrite is not a bad rewrite -- it is a
    different transform. A 5,936-character legal narrative cut to 312 characters is a *summary*,
    and scoring it as a paraphrase would attribute a large ΔNDCG to phrasing when the real cause
    was a decoding limit.
    """


class Generator(ABC):
    def __init__(self, config: GeneratorConfig, temperature: float, seed: int) -> None:
        self.config = config
        self.temperature = temperature
        self.seed = seed

    def _check_complete(self, completion: Completion) -> Completion:
        """Reject a completion the provider says stopped early.

        Only the finish reason is checked here, because it is the one authoritative signal a
        generator has. It matters most for thinking models: on Vertex `max_output_tokens` counts
        thinking tokens, so an unbounded reasoning trace can consume the whole budget and leave a
        fragment that reads like a deliberate short rewrite.

        Length is deliberately NOT checked here. A generator does not know the source text, and
        a purely textual heuristic misfires on technical content -- a valid HumanEval rewrite
        ends `MAD = average | x - x_mean |`, which no punctuation rule can distinguish from a
        cut-off sentence. That check belongs where the source is in hand; see
        `scripts/02_run_transform.py`.
        """
        if completion.finish_reason.upper() in {"MAX_TOKENS", "LENGTH"}:
            raise TruncatedCompletion(
                f"{self.config.id}: hit the output limit ({completion.finish_reason}); "
                "raise max_output_tokens or reduce thinking"
            )
        return completion

    @property
    def lineage(self) -> str:
        return self.config.lineage

    @abstractmethod
    def generate(self, prompt: str) -> Completion: ...


class VertexGenerator(Generator):
    """Gemini family via Vertex AI."""

    def __init__(self, config: GeneratorConfig, temperature: float, seed: int) -> None:
        super().__init__(config, temperature, seed)
        from google import genai

        self._genai = genai
        self._client = genai.Client(
            vertexai=True,
            project=_require_env("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def generate(self, prompt: str) -> Completion:
        config = self._genai.types.GenerateContentConfig(
            temperature=self.temperature,
            seed=self.seed,
            max_output_tokens=self.config.max_output_tokens,
        )
        # On Vertex, max_output_tokens counts thinking tokens for the 2.5 family, so an
        # unconstrained reasoning trace silently ate the budget and truncated 44 of 50
        # AILAStatutes rewrites mid-sentence. Pin the budget low; 2.5 Pro rejects 0 outright.
        if self.config.thinking_budget is not None:
            config.thinking_config = self._genai.types.ThinkingConfig(
                thinking_budget=self.config.thinking_budget
            )
        response = self._client.models.generate_content(
            model=self.config.id, contents=prompt, config=config
        )
        text = (response.text or "").strip()
        if not text:
            raise ValueError(f"{self.config.id}: empty completion (finish reason may be a filter)")
        finish = ""
        if response.candidates:
            finish = str(getattr(response.candidates[0], "finish_reason", "") or "")
        return self._check_complete(
            Completion(
                text=text,
                model_version=getattr(response, "model_version", "") or self.config.id,
                finish_reason=finish,
            )
        )


class NovitaGenerator(Generator):
    """Open models via Novita's OpenAI-compatible chat endpoint."""

    _BASE_URL = "https://api.novita.ai/v3/openai"

    def __init__(self, config: GeneratorConfig, temperature: float, seed: int) -> None:
        super().__init__(config, temperature, seed)
        from openai import OpenAI

        self._client = OpenAI(api_key=_require_env("NOVITA_API_KEY"), base_url=self._BASE_URL)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def generate(self, prompt: str) -> Completion:
        response = self._client.chat.completions.create(
            model=self.config.id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=self.config.max_output_tokens,
        )
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise ValueError(f"{self.config.id}: empty completion")
        return self._check_complete(Completion(
            text=text, model_version=response.model or self.config.id,
            finish_reason=str(choice.finish_reason or ""),
        ))


class VertexMaasGenerator(Generator):
    """Open models (Llama) via Vertex AI Model-as-a-Service.

    Vertex exposes Llama 3.3 70B through an OpenAI-shaped `openapi/chat/completions` path,
    authenticated with an ADC bearer token -- the same ADC as the Gemini route, but a different
    transport. Added because Novita's Llama endpoint went into sustained 429 `server_overload`
    mid-study; Vertex serves the same weights, so the lineage id is preserved and only the route
    changes. The ADC token expires roughly hourly, so it is refreshed lazily before it lapses.
    """

    def __init__(self, config: GeneratorConfig, temperature: float, seed: int) -> None:
        super().__init__(config, temperature, seed)
        import google.auth
        from google.auth.transport.requests import Request

        self._creds, adc_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._project = _require_env("GOOGLE_CLOUD_PROJECT") or adc_project
        self._location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        self._auth_request = Request()
        self._client = None
        self._refresh_client()

    def _refresh_client(self) -> None:
        from openai import OpenAI

        self._creds.refresh(self._auth_request)
        base_url = (
            f"https://{self._location}-aiplatform.googleapis.com/v1/"
            f"projects/{self._project}/locations/{self._location}/endpoints/openapi"
        )
        self._client = OpenAI(api_key=self._creds.token, base_url=base_url)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _call(self, prompt: str):
        """One API call, with exponential backoff for genuine API faults (rate limits, network)."""
        if not self._creds.valid:
            self._refresh_client()
        return self._client.chat.completions.create(
            model=self.config.api_model_name or self.config.id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=self.config.max_output_tokens,
        )

    def generate(self, prompt: str) -> Completion:
        # Vertex MaaS Llama intermittently returns an EMPTY completion on long inputs, which clears
        # on an immediate re-request. Retry empties tightly here (no backoff) instead of paying
        # `_call`'s exponential backoff -- that backoff is for real API faults, and applying it to
        # transient empties was the dominant cost, dragging throughput to ~16/min against a 0.35s
        # endpoint. A persistently empty query (4 immediate tries) raises, and the transform loop
        # keeps the original query text (a zero-ΔNDCG no-op). Six tries, not two or three: the
        # empty rate is bursty (near-zero for long stretches, then ~75% for minutes at a time), so
        # a few extra immediate re-requests keep fallbacks negligible through a bad patch.
        for _ in range(6):
            response = self._call(prompt)
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            if text:
                return self._check_complete(Completion(
                    text=text, model_version=response.model or self.config.id,
                    finish_reason=str(choice.finish_reason or ""),
                ))
        raise ValueError(f"{self.config.id}: empty completion")


class AnthropicGenerator(Generator):
    """Claude family via the Anthropic API.

    Note `temperature` is deliberately absent. It was removed on Claude Opus 4.7 and later and
    now returns a 400, so this route is sampled at the model's own default and the config records
    that via `supports_temperature: false`. `thinking` is left unset, which runs the model without
    extended thinking -- a paraphrase needs no reasoning trace, and omitting it keeps the
    generation fast and the output free of reasoning preamble.
    """

    def __init__(self, config: GeneratorConfig, temperature: float, seed: int) -> None:
        super().__init__(config, temperature, seed)
        import anthropic

        self._client = anthropic.Anthropic()

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def generate(self, prompt: str) -> Completion:
        kwargs = {"temperature": self.temperature} if self.config.supports_temperature else {}
        response = self._client.messages.create(
            model=self.config.id,
            max_tokens=self.config.max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise ValueError(f"{self.config.id}: refused ({response.stop_details})")

        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise ValueError(f"{self.config.id}: empty completion ({response.stop_reason})")
        return self._check_complete(Completion(
            text=text, model_version=response.model or self.config.id,
            finish_reason=str(response.stop_reason or ""),
        ))


class OpenAIGenerator(Generator):
    """GPT family via the OpenAI API."""

    def __init__(self, config: GeneratorConfig, temperature: float, seed: int) -> None:
        super().__init__(config, temperature, seed)
        from openai import OpenAI

        self._client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def generate(self, prompt: str) -> Completion:
        kwargs = {"temperature": self.temperature} if self.config.supports_temperature else {}
        response = self._client.chat.completions.create(
            model=self.config.id,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError(f"{self.config.id}: empty completion")
        return Completion(text=text, model_version=response.model or self.config.id)


_ROUTES: dict[Provider, type[Generator]] = {
    Provider.VERTEX: VertexGenerator,
    Provider.VERTEX_MAAS: VertexMaasGenerator,
    Provider.NOVITA: NovitaGenerator,
    Provider.ANTHROPIC: AnthropicGenerator,
    Provider.OPENAI: OpenAIGenerator,
}


def get_generator(config: GeneratorConfig, temperature: float, seed: int) -> Generator:
    if config.provider not in _ROUTES:
        raise KeyError(
            f"{config.id}: no generator route for provider {config.provider}; "
            f"available: {', '.join(p.value for p in _ROUTES)}"
        )
    return _ROUTES[config.provider](config, temperature, seed)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; see .env.example")
    return value
