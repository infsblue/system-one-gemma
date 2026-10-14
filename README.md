# system-one-gemma

An open-source Python engine for fast, deterministic, "no-chat" System 1 decision-making and guardrails using Gemma.

## About

**system-one-gemma** is a lightweight, local-first decision engine inspired by System 1 cognitive processing. Designed to run entirely on local hardware, it bypasses traditional multi-turn conversational chat LLMs to deliver deterministic, type-safe decisions, classifications, and confidence scoring in milliseconds.

By leveraging Google's **Gemma 1.1 2B IT** model via local GGUF inference, constrained sampling (grammars/JSON Schema), and raw token probability evaluation (`logprobs`), it acts as an ultra-fast guardrail, router, and classification layer without sending context to third-party cloud APIs.

## Model

* **Model Name:** Gemma 1.1 2B IT (google/gemma-1.1-2b-it-GGUF)
* **Official Weights File:** `2b_it_v1p1.gguf` (~10 GB, FP32/F16 precision)
* **Execution Engine:** Local execution via `llama-cpp-python` with full or partial GPU VRAM offloading
* **Licensing:** Governed by Google Gemma Terms of Use

## Requirements

### Hardware Specs

* **CPU:** 6-core x86_64 Processor (e.g., Intel Core i7-10750H)
* **RAM:** 16 GB minimum (32 GB+ recommended for large context buffers)
* **GPU:** NVIDIA Dedicated GPU with 6 GB VRAM (e.g., GeForce RTX 2060) or higher

### Software Dependencies

* **OS:** Linux (Debian 13 / Ubuntu 22.04+ recommended)
* **Python Version:** Python 3.10+
* **Drivers:** NVIDIA Proprietary Drivers + CUDA Toolkit (`nvidia-smi` operational)

### Python Dependencies (`requirements.txt`)

* llama-cpp-python>=0.2.75
* pydantic>=2.0.0
* python-dotenv>=1.0.1

## Decision Primitives & Capabilities

`system-one-gemma` exposes 3 deterministic decision primitives that enforce structured JSON outputs while extracting mathematical confidence scores directly from model logprobs (`math.exp(logprob)`).

### 1. `Noul` (Probabilistic Boolean Guardrail)

Evaluates a single state against a binary question and calculates the exact model confidence.

* **Function:** Evaluates inputs (such as user prompts or generated SQL) against binary safety rules.
* **Example Use Case:** Verifying whether a query string contains malicious input or SQL injection attempts before sending it to a database.
* **Output:** Returns a boolean value (`True`/`False`) paired with a calibrated confidence score (e.g., `False` with 99.4% confidence).

### 2. `Choice` (Constrained Categorical Selection)

Forces the engine to pick exactly one option from a predefined list without conversational fluff.

* **Function:** Maps unstructured incoming text to a fixed category or execution path.
* **Example Use Case:** Classifying user request intent (e.g., routing requests between read-only SQL execution, schema inspection, data mutation, or off-topic queries).
* **Output:** Returns the selected category string alongside the model's categorical confidence (e.g., `READ_ONLY_SQL` with 96.8% confidence).

### 3. `Score` (Discretization & Ordinal Rating)

Maps unstructured inputs onto a strict numerical scale with probability validation.

* **Function:** Evaluates qualitative inputs and converts them into discrete numerical values within a bounded range.
* **Example Use Case:** Assessing the risk severity level of a operation on a scale from 1 (safe) to 5 (critical mutation).
* **Output:** Returns an integer score within the specified bounds along with the calculated statistical confidence (e.g., Risk Level `5` with 98.1% confidence).

## Acceptance Criteria

To ensure reliable, low-latency, and production-ready operation, the pipeline enforces the following criteria:

1. **Zero-Chat Determinism:**
* Enforces strict C++ GBNF grammars / JSON Schemas via `llama-cpp-python` to prevent free-text or conversational output.


2. **Real-Calibrated Confidence:**
* Calculates output confidence scores using token-level log probabilities rather than self-reported LLM verbalization.


3. **Sub-100ms Execution Bounds:**
* Optimized for single-pass evaluation using Gemma 1.1 2B IT, achieving low-latency decision-making on 6 GB VRAM GPUs.


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
* `python-dotenv`: BSD 3-Clause License

### Model Weights Notice

* **Gemma:** Model weights, parameters, and architecture belong to Google LLC and are governed by the Gemma Terms of Use and Gemma Prohibited Use Policy.
* This repository does not host, bundle, or re-distribute model weights. Users must separately accept Google's terms and acquire `.gguf` files directly from official sources (`google/gemma-1.1-2b-it-GGUF`).