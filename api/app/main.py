import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import config
from .engine import Choice, DecisionResult, Noul, Score, load_model
from .scenarios import SCENARIOS, Scenario

DESCRIPTION = """
**system-one-gemma** is a local decision engine. It makes *one fast, constrained decision*
per request using a local CodeGemma model (no cloud calls).

## The three primitives

| Primitive | Question it answers | Output |
|---|---|---|
| **Noul** | A yes/no question about some input | `true` / `false` |
| **Choice** | Which ONE of these categories fits? | one of your options |
| **Score** | How high on a numeric scale? | an integer in `[minimum, maximum]` |

## How to use this page

1. **Tests** - six preconfigured experiments (one *positive* and one *negative* per
   primitive). Click *Try it out* -> *Execute*; no input needed. The response shows the
   material that was sent, what was expected, what the model answered, and whether it passed.
2. **Try it yourself** - one endpoint per primitive where you provide your own context.

## Reading `confidence`

`confidence` is the model's probability for the chosen answer, **renormalized over the
answers the schema allows** (e.g. `true` vs `false`). `1.0` means the model put essentially
no weight on the alternatives; `0.5` on a yes/no question means a coin flip. It is **not** a
calibrated guarantee that the answer is correct, so do not use it as the only safety control.

## Notes

* Output is schema-constrained: the model physically cannot answer outside the allowed values.
* Decoding is greedy with a fixed seed, so the same input gives the same answer.
* Inference runs one request at a time (the model is shared); expect seconds per call on CPU.
"""

TAGS = [
    {
        "name": "Tests - Noul",
        "description": "Boolean guardrail. **Positive**: the risky condition is present "
        "(expect `true`). **Negative**: it is absent (expect `false`).",
    },
    {
        "name": "Tests - Choice",
        "description": "Constrained category selection. **Positive**: an in-domain request "
        "(expect `READ_ONLY_SQL`). **Negative**: an unrelated one (expect `OFF_TOPIC`).",
    },
    {
        "name": "Tests - Score",
        "description": "Ordinal rating from 1 (safe) to 5 (critical). **Positive**: dangerous "
        "SQL (expect >= 4). **Negative**: read-only SQL (expect <= 2).",
    },
    {
        "name": "Try it yourself",
        "description": "Provide your own context and get the engine's decision.",
    },
    {"name": "System", "description": "Service status."},
]


def interpret_confidence(confidence: float) -> str:
    if confidence >= 0.9:
        return (
            "High: the model put almost all of its probability on this answer "
            "versus the allowed alternatives."
        )
    if confidence >= 0.6:
        return "Moderate: this answer is favored, but the alternatives had real weight."
    return "Low: the model was torn between several allowed answers; treat as uncertain."


class NoulRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="A binary (yes/no) question the model must answer about the state.",
        examples=["Does this SQL contain a destructive operation?"],
    )
    state: str = Field(
        min_length=1,
        description="The input to evaluate (a prompt, generated SQL, ...). "
        "It is treated as untrusted data, never as instructions.",
        examples=["DELETE FROM customers;"],
    )


class ChoiceRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="What the choice represents.",
        examples=["Route this user request to the correct handler."],
    )
    input: str = Field(
        min_length=1,
        description="The text to classify (untrusted data).",
        examples=["List all columns of the orders table."],
    )
    options: list[str] = Field(
        min_length=2,
        description="Allowed categories. The model must return exactly one of them.",
        examples=[["READ_ONLY_SQL", "SCHEMA_INSPECTION", "DATA_MUTATION", "OFF_TOPIC"]],
    )


class ScoreRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="What the score measures, including what the low and high ends mean.",
        examples=[
            "Rate the risk severity of this SQL, where 1 is safe and 5 is a critical mutation."
        ],
    )
    input: str = Field(
        min_length=1,
        description="The text to rate (untrusted data).",
        examples=["UPDATE accounts SET balance = 0;"],
    )
    minimum: int = Field(1, description="Lowest allowed score (inclusive).")
    maximum: int = Field(5, description="Highest allowed score (inclusive).")


class DecisionResponse(BaseModel):
    primitive: Literal["noul", "choice", "score"] = Field(
        description="Which primitive produced this decision."
    )
    value: bool | str | int = Field(
        description="The decision: bool for noul, category for choice, integer for score."
    )
    confidence: float = Field(
        description="Probability of the chosen answer, renormalized over the allowed "
        "answers (0-1). Not a calibrated correctness guarantee."
    )
    confidence_reading: str = Field(description="Plain-language reading of `confidence`.")
    token_count: int = Field(description="Number of output tokens that made up the decision.")


class TestMaterial(BaseModel):
    question: str = Field(description="Evaluation question sent to the model.")
    input: str = Field(description="Input the model evaluated.")
    options: list[str] | None = Field(None, description="Allowed categories (choice only).")
    scale: str | None = Field(None, description="Allowed score range (score only).")


class TestReport(BaseModel):
    test: str = Field(description="Test identifier.")
    primitive: Literal["noul", "choice", "score"]
    kind: Literal["positive", "negative"] = Field(
        description="positive: the condition holds / the request is in-domain. "
        "negative: the condition does not hold / the request is out-of-domain."
    )
    title: str
    purpose: str = Field(description="What this test demonstrates and why the expectation holds.")
    material: TestMaterial = Field(description="Exactly what was sent to the model.")
    expected: str = Field(description="What a correct answer looks like.")
    actual: DecisionResponse = Field(description="What the engine actually returned.")
    passed: bool = Field(description="True when the actual decision satisfies the expectation.")
    verdict: str = Field(description="One-sentence human summary of the outcome.")


def build_decision(primitive: str, result: DecisionResult[Any]) -> DecisionResponse:
    return DecisionResponse(
        primitive=primitive,  # type: ignore[arg-type]
        value=result.value,
        confidence=result.confidence,
        confidence_reading=interpret_confidence(result.confidence),
        token_count=result.token_count,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(config.MODEL_PATH)
    if not model_path.is_file():
        raise RuntimeError(f"model not found: {model_path}")
    app.state.llm = load_model(str(model_path), config.N_GPU_LAYERS, config.N_CTX)
    app.state.lock = threading.Lock()
    yield


app = FastAPI(
    title="system-one-gemma",
    summary="Local decision primitives: Noul, Choice and Score.",
    description=DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS,
    lifespan=lifespan,
)


def _run(request: Request, primitive: str, call) -> DecisionResponse:
    llm = request.app.state.llm
    try:
        # llama.cpp contexts are not thread-safe: serialize inference
        with request.app.state.lock:
            result = call(llm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"model produced an unusable decision: {exc}")
    return build_decision(primitive, result)


def _run_scenario(name: str, request: Request) -> TestReport:
    sc: Scenario = SCENARIOS[name]
    if sc.primitive == "noul":
        decision = _run(request, "noul", lambda llm: Noul(llm).evaluate(sc.input_text, sc.question))
    elif sc.primitive == "choice":
        decision = _run(
            request, "choice", lambda llm: Choice(llm).select(sc.input_text, sc.options, sc.question)
        )
    else:
        decision = _run(
            request,
            "score",
            lambda llm: Score(llm).evaluate(sc.input_text, sc.question, sc.minimum, sc.maximum),
        )
    passed = sc.check(decision.value)
    verdict = (
        f"PASS: expected {sc.expected} and the engine answered {decision.value!r} "
        f"(confidence {decision.confidence:.1%})."
        if passed
        else f"FAIL: expected {sc.expected} but the engine answered {decision.value!r} "
        f"(confidence {decision.confidence:.1%}). This shows the model can be wrong; "
        "do not rely on it alone."
    )
    return TestReport(
        test=name,
        primitive=sc.primitive,
        kind=sc.kind,
        title=sc.title,
        purpose=sc.purpose,
        material=TestMaterial(
            question=sc.question,
            input=sc.input_text,
            options=sc.options or None,
            scale=f"{sc.minimum} to {sc.maximum}" if sc.primitive == "score" else None,
        ),
        expected=sc.expected,
        actual=decision,
        passed=passed,
        verdict=verdict,
    )


def _register_test(name: str, tag: str) -> None:
    sc = SCENARIOS[name]

    def endpoint(request: Request) -> TestReport:
        return _run_scenario(name, request)

    endpoint.__name__ = name.replace("-", "_")
    app.post(
        f"/tests/{sc.primitive}/{sc.kind}",
        response_model=TestReport,
        tags=[tag],
        summary=sc.title,
        description=(
            f"{sc.purpose}\n\n"
            f"**Preconfigured material (nothing to fill in):**\n\n"
            f"* Question: `{sc.question}`\n"
            f"* Input: `{sc.input_text}`\n"
            + (f"* Options: `{', '.join(sc.options)}`\n" if sc.options else "")
            + f"* Expected: **{sc.expected}**\n\n"
            "Click *Execute*. The response reports what the engine answered, its "
            "confidence, and whether the test passed."
        ),
    )(endpoint)


for _name, _sc in SCENARIOS.items():
    _register_test(_name, f"Tests - {_sc.primitive.capitalize()}")


@app.post(
    "/noul",
    response_model=DecisionResponse,
    tags=["Try it yourself"],
    summary="Noul: ask a yes/no question about your input",
    description="Evaluates `state` against a binary `question` and returns `true`/`false` "
    "with the model's confidence. Typical use: is this SQL/prompt malicious or destructive?",
)
def noul(req: NoulRequest, request: Request) -> DecisionResponse:
    return _run(request, "noul", lambda llm: Noul(llm).evaluate(req.state, req.question))


@app.post(
    "/choice",
    response_model=DecisionResponse,
    tags=["Try it yourself"],
    summary="Choice: pick exactly one category for your input",
    description="Forces the model to return exactly one of `options`, with no extra text. "
    "Typical use: route a request to the right handler.",
)
def choice(req: ChoiceRequest, request: Request) -> DecisionResponse:
    return _run(
        request, "choice", lambda llm: Choice(llm).select(req.input, req.options, req.question)
    )


@app.post(
    "/score",
    response_model=DecisionResponse,
    tags=["Try it yourself"],
    summary="Score: rate your input on an integer scale",
    description="Returns an integer between `minimum` and `maximum` (inclusive). Describe "
    "what the low and high ends mean in `question` for best results.",
)
def score(req: ScoreRequest, request: Request) -> DecisionResponse:
    return _run(
        request,
        "score",
        lambda llm: Score(llm).evaluate(req.input, req.question, req.minimum, req.maximum),
    )


@app.get("/health", tags=["System"], summary="Service and model status")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "model": Path(config.MODEL_PATH).name}
