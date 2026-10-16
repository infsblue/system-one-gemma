> [!WARNING]
> **Proof of Concept — under constant change and research.**
> This repository is an early-stage proof of concept and is evolving continuously. The documentation below describes the **initial requirements** of the project and may not reflect the current state of the code, the model, or the setup. Dependencies, endpoints, hardware requirements, and acceptance criteria may differ from what is written here. When in doubt, the source code is the reference.
>
> This project uses a System-1-inspired, single-pass, schema-constrained, deterministic, short-form token-generative architecture. It is designed to produce fast constrained decisions, not guaranteed-correct answers. Outputs are probabilistic and may be wrong; validate critical decisions with deterministic rules and/or human review.

# system-one-gemma

An open-source, local Python engine for schema-constrained System-1-inspired decisions and guardrails using Gemma.

## About

**system-one-gemma** is a lightweight, local-first decision engine inspired by System 1 cognitive processing. It uses single-pass model inference to return schema-constrained decisions, classifications, and raw token likelihoods without sending context to third-party cloud APIs. Output constraints and deterministic sampling do not guarantee that a model's decision is correct.

By leveraging Google's **CodeGemma 1.1 7B IT** model via local GGUF inference, constrained sampling (grammars/JSON Schema), and raw token probability evaluation (`logprobs`), it acts as a local guardrail, router, and classification layer without sending context to third-party cloud APIs.

## Model

* **Model Name:** CodeGemma 1.1 7B IT
* **Weights File:** `models/codegemma-1.1-7b-it-f16.gguf` (~16 GB, F16), mounted read-only into the container (override with `MODEL_FILE` in `.env`)
* **Execution Engine:** `llama-cpp-python` (CPU build) inside Docker
* **Licensing:** Governed by Google Gemma Terms of Use

## Requirements

Everything else (Python, dependencies, llama.cpp) is packaged in the Docker image.

### Hardware Specs (minimum to run the model on CPU)

* **CPU:** any x86_64 CPU (4+ cores recommended; a call takes seconds)
* **RAM:** 24 GB available to Docker (~16 GB weights + context and logits buffers)
* **Disk:** ~17 GB for the model file plus the image

### Software Dependencies

* **Docker** with Docker Compose v2

### Python Dependencies (inside the image, `api/requirements.txt`)

* llama-cpp-python>=0.3.0
* pydantic>=2.0.0
* fastapi>=0.110
* uvicorn[standard]>=0.29

## Decision Primitives & Capabilities

`api/app/engine.py` implements three decision primitives. Each uses JSON Schema-constrained decoding,
greedy sampling, and a fixed seed, and returns a `DecisionResult` with the selected value,
its normalized model confidence, and the number of output tokens used for that value.

The reported confidence is the probability of the chosen first decision token, renormalized
over the tokens that can start an allowed value (e.g. `true`/`false`), because raw logprobs
are taken before grammar masking. For multi-token values (long category names) this is a
first-token approximation. It is **not a calibrated probability that the answer is correct** and should not be used as a
standalone safety guarantee. Calibrating confidence requires a representative labeled
evaluation set, which is not included here.

### Running the API

The engine is served by FastAPI (`api/app`) in Docker:

```bash
cp .env.example .env        # optional overrides
docker compose up --build   # model is mounted read-only from ./models
```

Open **http://localhost:8000/docs** (Swagger). Endpoints:

| Group | Endpoint | What it does |
|---|---|---|
| Tests - Noul | `POST /tests/noul/positive`, `/tests/noul/negative` | Destructive SQL is flagged / harmless SQL is not |
| Tests - Choice | `POST /tests/choice/positive`, `/tests/choice/negative` | In-domain request routed to `READ_ONLY_SQL` / unrelated request to `OFF_TOPIC` |
| Tests - Score | `POST /tests/score/positive`, `/tests/score/negative` | `DROP`/`TRUNCATE` scores >= 4 / a `SELECT` scores <= 2 |
| Try it yourself | `POST /noul`, `/choice`, `/score` | Send your own context |
| System | `GET /health` | Status and model name |

The test endpoints need no input: the material lives in `api/app/scenarios.py`, and each
response shows what was sent, what was expected, what the model answered, and pass/fail.
The tests report real model behavior; a FAIL is a legitimate outcome, not an API error.

The model's judgment remains probabilistic: validate safety-critical decisions
with deterministic rules and/or human review.

### 1. `Noul` (Probabilistic Boolean Guardrail)

Evaluates a state against a binary question and returns a boolean decision with raw token likelihood.

* **Function:** Evaluates inputs (such as user prompts or generated SQL) against binary safety rules.
* **Example Use Case:** Verifying whether a query string contains malicious input or SQL injection attempts before sending it to a database.
* **Output:** Returns a boolean value (`true`/`false`) with the raw likelihood of the generated decision tokens.

### 2. `Choice` (Constrained Categorical Selection)

Forces the engine to pick exactly one option from a predefined list without conversational fluff.

* **Function:** Maps unstructured incoming text to a fixed category or execution path.
* **Example Use Case:** Classifying user request intent (e.g., routing requests between read-only SQL execution, schema inspection, data mutation, or off-topic queries).
* **Output:** Returns the selected category string with the raw likelihood of the generated decision tokens.

### 3. `Score` (Discretization & Ordinal Rating)

Maps unstructured inputs onto a strict numerical scale with range validation.

* **Function:** Evaluates qualitative inputs and converts them into discrete numerical values within a bounded range.
* **Example Use Case:** Assessing the risk severity level of a operation on a scale from 1 (safe) to 5 (critical mutation).
* **Output:** Returns an integer score within the specified bounds with the raw likelihood of the generated decision tokens.

## Acceptance Criteria

To ensure reliable, low-latency, and production-ready operation, the pipeline enforces the following criteria:

1. **Constrained Output:**
* Enforces JSON Schema constraints through `llama-cpp-python` to restrict decisions to the requested type, categories, or numeric range.


2. **Token Likelihood:**
* Reports the likelihood of the generated decision value using token-level log probabilities. This is not calibrated confidence; calibration against labeled data is future work.


3. **Local Inference:**
* Runs on the configured local CPU/GPU. Latency depends on hardware, model configuration, and input length; sub-100ms execution is not guaranteed.


4. **Sandwich Pattern Integration:**
* Built to serve as pre-validation (input filtering) and post-validation (output inspection) guardrails for heavy System 2 engines (e.g., `text2sql-postgres-gemma`).


5. **Licensing Isolation:**
* Codebase distributed under the BSD 2-Clause License.
* Model weights (`*.gguf`) are strictly excluded from Git tracking via `.gitignore`.

## Licensing & Dependencies

### Source Code License

This project's original source code is licensed under the **BSD 2-Clause License (BSD Simplified)**. See the `LICENSE` file for the full legal text.

### Third-Party Dependencies Licenses

All direct dependencies used in this project are compatible with permissive open-source licensing:

* `llama-cpp-python`: MIT License
* `pydantic`: MIT License
* `fastapi`: MIT License
* `uvicorn`: BSD 3-Clause License

### Model Weights Notice

* **Gemma:** Model weights, parameters, and architecture belong to Google LLC and are governed by the Gemma Terms of Use and Gemma Prohibited Use Policy.
* This repository does not host, bundle, or re-distribute model weights. Users must separately accept Google's terms and acquire `.gguf` files directly from official sources (CodeGemma 1.1 7B IT).
