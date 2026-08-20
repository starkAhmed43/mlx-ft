from __future__ import annotations

import pytest


@pytest.fixture()
def sample_record() -> dict[str, object]:
    return {
        "id": "fixture-1",
        "query": " Find the weather. ",
        "tools": [
            {
                "name": "weather.get",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "enum": ["Paris", "Delhi"]},
                        "days": {"type": "integer"},
                    },
                    "required": ["city"],
                },
            }
        ],
        "answers": [{"name": "weather.get", "arguments": {"city": "Paris"}}],
    }
