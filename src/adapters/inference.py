"""LlamaCpp-backed inference adapter implementing InferencePort."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse
from domain.ports import InferencePort
from adapters.prompt import build_prompt, parse_response

if TYPE_CHECKING:
    import llama_cpp as _llama_cpp

log = logging.getLogger(__name__)

# GBNF grammar that forces the sampler to emit a valid InferenceResponse JSON.
# Each optional field appears AT MOST ONCE, preventing the model from looping on
# repeated keys (e.g. ,"confidence":0.90,"confidence":0.90,...) until max_tokens.
_RESPONSE_GBNF = (
    'root ::= "{" ws "\\"stat\\"" ws ":" ws stat-val ws "," ws "\\"action\\"" ws ":" ws action-val target-part confidence-part ws "}"\n'
    "ws ::= [ \\t\\n]*\n"
    'stat-val ::= "\\"hunger\\"" | "\\"tiredness\\"" | "\\"boredom\\"" | "\\"social\\"" | "\\"toilet\\""\n'
    'action-val ::= "\\"EAT\\"" | "\\"DRINK\\"" | "\\"PLAY\\"" | "\\"FETCH\\"" | "\\"SLEEP\\"" | "\\"SOCIAL\\"" | "\\"FOLLOW\\"" | "\\"TOILET\\"" | "\\"IDLE\\"" | "\\"EXPLORE\\""\n'
    'target-part ::= (ws "," ws "\\"target_object_id\\"" ws ":" ws (id-str | "null"))?\n'
    'confidence-part ::= (ws "," ws "\\"confidence\\"" ws ":" ws number)?\n'
    'id-str ::= "\\"" [-a-zA-Z0-9_]* "\\""\n'
    'number ::= [0-9]+ ("." [0-9]+)?\n'
)

# Actions that require a target object and the scene types they must come from.
_ACTION_TARGET_TYPES: dict[Action, set[str]] = {
    Action.EAT: {"bowl"},
    Action.DRINK: {"bowl"},
    Action.PLAY: {"toy"},
    Action.FETCH: {"toy"},
    Action.SLEEP: {"bed"},
    Action.SOCIAL: {"player", "pet"},
    Action.FOLLOW: {"player", "pet"},
}


def _import_llama_cpp() -> Any:
    try:
        import llama_cpp
        return llama_cpp
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is not installed. "
            "Install it with: pip install 'aipet-llm-api[inference]'"
        ) from exc


class LlamaCppInferenceAdapter(InferencePort):
    """InferencePort implementation backed by a GGUF-quantised model via llama-cpp-python.

    The model is loaded lazily on the first call to ``infer`` so that
    construction is cheap and test set-up does not require a real model file.
    """

    def __init__(self, model_path: str, context_size: int = 2048) -> None:
        self._model_path = model_path
        self._context_size = context_size
        self._llm: Any = None
        self._grammar: Any = None
        self._llama_cpp: Any = None

    def _get_llama_cpp(self) -> Any:
        if self._llama_cpp is None:
            self._llama_cpp = _import_llama_cpp()
            try:
                self._grammar = self._llama_cpp.LlamaGrammar.from_string(_RESPONSE_GBNF)
            except Exception as exc:
                log.warning("Grammar-constrained sampling unavailable: %s", exc)
                self._grammar = None
        return self._llama_cpp

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Instantiate and return the Llama model (called once, lazily)."""
        llama_cpp = self._get_llama_cpp()
        log.info("Loading GGUF model from %s", self._model_path)
        return llama_cpp.Llama(
            model_path=self._model_path,
            n_ctx=self._context_size,
            verbose=False,
        )

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = self._load_model()
        return self._llm

    def _ensure_target(self, response: InferenceResponse, request: InferenceRequest) -> InferenceResponse:
        """Guarantee the response has a valid target for actions that require one.

        If the model omitted the target or returned one with the wrong object type,
        replace it with the closest scene object whose type satisfies the action.
        """
        required_types = _ACTION_TARGET_TYPES.get(response.action)
        if not required_types:
            return response

        valid_ids = {o.id for o in request.scene.objects if o.type in required_types}
        if response.target_object_id in valid_ids:
            return response

        candidates = [o for o in request.scene.objects if o.type in required_types]
        if not candidates:
            return response
        closest = min(candidates, key=lambda o: o.distance)
        return InferenceResponse(
            action=response.action,
            target_object_id=closest.id,
            confidence=response.confidence,
        )

    # ------------------------------------------------------------------
    # InferencePort implementation
    # ------------------------------------------------------------------

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference and return a structured response.

        On any recoverable error (model loading failure, malformed output, etc.)
        a warning is logged and ``InferenceResponse(action=Action.IDLE)`` is
        returned so the pet remains in a safe, neutral state.
        """
        fallback = InferenceResponse(action=Action.IDLE, target_object_id=None)

        try:
            prompt = build_prompt(request)
            llm = self._get_llm()
            completion: Any = llm(
                prompt,
                max_tokens=128,
                temperature=0.1,
                stop=["```"],
                grammar=self._grammar,
            )
            log.info(f"LLM Response: {completion}")
            raw_text: str = completion["choices"][0]["text"]
            response = parse_response(raw_text)
            return self._ensure_target(response, request)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Inference failed, returning IDLE fallback. Reason: %s",
                exc,
                exc_info=True,
            )
            return fallback
