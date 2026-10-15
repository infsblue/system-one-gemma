import json
import math
import unittest

import numpy as np

from app.engine import Choice, Noul, Score


class FakeLlama:
    def __init__(self, content: str, decision_token: str) -> None:
        self.content = content
        start = content.encode("utf-8").index(decision_token.encode("utf-8"))
        end = start + len(decision_token.encode("utf-8"))
        encoded = content.encode("utf-8")
        self.response = {
            "choices": [
                {
                    "message": {"content": content},
                    "logprobs": {
                        "content": [
                            {
                                "token": encoded[:start].decode("utf-8"),
                                "bytes": list(encoded[:start]),
                                "logprob": -4.0,
                            },
                            {
                                "token": decision_token,
                                "bytes": list(encoded[start:end]),
                                "logprob": np.float32(-0.25),
                                "top_logprobs": [
                                    {"token": decision_token, "logprob": -0.25},
                                    {"token": "true" if decision_token == "false" else "false", "logprob": -1.5},
                                    {"token": "The", "logprob": -0.1},
                                ],
                            },
                            {
                                "token": encoded[end:].decode("utf-8"),
                                "bytes": list(encoded[end:]),
                                "logprob": -4.0,
                            },
                        ]
                    },
                }
            ]
        }
        self.call: dict[str, object] = {}

    def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
        self.call = kwargs
        return self.response


class DecisionPrimitiveTests(unittest.TestCase):
    def test_noul_returns_boolean_and_value_likelihood(self) -> None:
        llm = FakeLlama('{"decision":false}', "false")

        result = Noul(llm).evaluate("DROP TABLE users", "Is this destructive SQL?")

        self.assertFalse(result.value)
        expected = math.exp(-0.25) / (math.exp(-0.25) + math.exp(-1.5))
        self.assertAlmostEqual(result.confidence, expected, places=6)
        self.assertEqual(result.token_count, 1)
        self.assertTrue(llm.call["logprobs"])
        self.assertEqual(llm.call["temperature"], 0.0)

    def test_choice_schema_limits_allowed_categories(self) -> None:
        llm = FakeLlama('{"decision":"READ_ONLY_SQL"}', '"READ_ONLY_SQL"')
        options = ["READ_ONLY_SQL", "DATA_MUTATION"]

        result = Choice(llm).select("List orders", options)

        self.assertEqual(result.value, "READ_ONLY_SQL")
        response_format = llm.call["response_format"]
        self.assertIsInstance(response_format, dict)
        schema = response_format["schema"]
        self.assertEqual(
            schema["properties"]["decision"]["enum"], options
        )

    def test_score_schema_and_result_enforce_bounds(self) -> None:
        llm = FakeLlama('{"decision":5}', "5")

        result = Score(llm).evaluate("Drop all rows", "Rate risk", 1, 5)

        self.assertEqual(result.value, 5)
        response_format = llm.call["response_format"]
        self.assertIsInstance(response_format, dict)
        decision_schema = response_format["schema"]["properties"]["decision"]
        self.assertEqual(decision_schema["minimum"], 1)
        self.assertEqual(decision_schema["maximum"], 5)

    def test_invalid_choice_options_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            Choice(FakeLlama('{"decision":"A"}', '"A"')).select(
                "input", ["A", "A"]
            )

    def test_invalid_score_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum"):
            Score(FakeLlama('{"decision":1}', "1")).evaluate(
                "input", "question", 5, 1
            )

    def test_cli_result_serializes_as_json(self) -> None:
        result = Noul(FakeLlama('{"decision":true}', "true")).evaluate(
            "safe", "Is this safe?"
        )

        self.assertEqual(json.loads(json.dumps(result.__dict__))["value"], True)


if __name__ == "__main__":
    unittest.main()
