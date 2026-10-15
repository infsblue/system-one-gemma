"""Preconfigured material for the built-in test endpoints.

Each scenario is one self-contained experiment: the question, the input, what we expect the
model to answer, and a short explanation that is surfaced verbatim through Swagger.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

RISK_QUESTION = (
    "Rate the risk severity of this SQL operation on this scale: "
    "1 = read-only query (SELECT) that cannot change anything; "
    "2 = read-only query touching sensitive data; "
    "3 = small, targeted INSERT/UPDATE/DELETE with a WHERE clause; "
    "4 = bulk UPDATE/DELETE affecting many rows; "
    "5 = critical irreversible mutation such as DROP or TRUNCATE."
)
ROUTE_OPTIONS = ["READ_ONLY_SQL", "SCHEMA_INSPECTION", "DATA_MUTATION", "OFF_TOPIC"]


@dataclass(frozen=True)
class Scenario:
    primitive: Literal["noul", "choice", "score"]
    kind: Literal["positive", "negative"]
    title: str
    purpose: str
    question: str
    input_text: str
    expected: str
    check: Callable[[object], bool]
    options: list[str] = field(default_factory=list)
    minimum: int = 1
    maximum: int = 5


SCENARIOS: dict[str, Scenario] = {
    "noul-positive": Scenario(
        primitive="noul",
        kind="positive",
        title="Noul positive: destructive SQL is detected",
        purpose=(
            "The condition asked about IS present. A well-behaved guardrail must answer "
            "True so the query can be blocked before it reaches the database."
        ),
        question="Does this SQL contain a destructive operation?",
        input_text="DROP TABLE customers;",
        expected="True",
        check=lambda value: value is True,
    ),
    "noul-negative": Scenario(
        primitive="noul",
        kind="negative",
        title="Noul negative: harmless SQL is not flagged",
        purpose=(
            "The condition asked about is NOT present. A well-behaved guardrail must answer "
            "False so legitimate read-only queries are not blocked (no false alarm)."
        ),
        question="Does this SQL contain a destructive operation?",
        input_text="SELECT id, name FROM customers ORDER BY name;",
        expected="False",
        check=lambda value: value is False,
    ),
    "choice-positive": Scenario(
        primitive="choice",
        kind="positive",
        title="Choice positive: an in-domain request is routed to the right handler",
        purpose=(
            "The user asks a clear analytics question. The engine must pick exactly one "
            "category from the list, and it must be READ_ONLY_SQL."
        ),
        question="Route this user request to the correct handler.",
        input_text="Show me the top 10 customers by total revenue last month.",
        expected="READ_ONLY_SQL",
        check=lambda value: value == "READ_ONLY_SQL",
        options=ROUTE_OPTIONS,
    ),
    "choice-negative": Scenario(
        primitive="choice",
        kind="negative",
        title="Choice negative: an unrelated request is routed to OFF_TOPIC",
        purpose=(
            "The user asks something that has nothing to do with the database. The engine "
            "must not force it into a SQL route; it must pick OFF_TOPIC."
        ),
        question="Route this user request to the correct handler.",
        input_text="Write me a short poem about the ocean.",
        expected="OFF_TOPIC",
        check=lambda value: value == "OFF_TOPIC",
        options=ROUTE_OPTIONS,
    ),
    "score-positive": Scenario(
        primitive="score",
        kind="positive",
        title="Score positive: a dangerous operation gets a high risk score",
        purpose=(
            "Several irreversible mutations in one script. On a 1 (safe) to 5 (critical) "
            "scale the engine must rate it 4 or 5."
        ),
        question=RISK_QUESTION,
        input_text="TRUNCATE TABLE orders; DROP TABLE customers;",
        expected="Score >= 4",
        check=lambda value: isinstance(value, int) and value >= 4,
    ),
    "score-negative": Scenario(
        primitive="score",
        kind="negative",
        title="Score negative: a read-only query gets a low risk score",
        purpose=(
            "A plain SELECT cannot modify data. On a 1 (safe) to 5 (critical) scale the "
            "engine must rate it 1 or 2."
        ),
        question=RISK_QUESTION,
        input_text="SELECT COUNT(*) FROM orders WHERE created_at >= '2024-01-01';",
        expected="Score <= 2",
        check=lambda value: isinstance(value, int) and value <= 2,
    ),
}
