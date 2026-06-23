# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Prajjwal Chittori

"""Dataset builders for the two benchmarks.

Both return a DataFrame with the column contract the runner and scorers share:
``context, question, answer_prefix, answer, task, max_new_tokens``. ``context`` is the long prompt KV
compression acts on; ``answer`` is a JSON reference the matching scorer reads.
"""

import json

import pandas as pd

CODE_EXEC_MAX_NEW_TOKENS = 512
TOOL_CALL_MAX_NEW_TOKENS = 128

COLUMNS = ["context", "question", "answer_prefix", "answer", "task", "max_new_tokens"]


# --------------------------------------------------------------------------------------------------
# code_exec — HumanEval: the model completes a function signature + docstring.
# --------------------------------------------------------------------------------------------------


def build_code_exec_df(limit: int | None = None) -> pd.DataFrame:
    """Build the code_exec DataFrame from HumanEval.

    Requires `datasets`. The prompt (function signature + docstring) is the `context`; the reference
    unit tests + entry point are carried in `answer` (JSON) for score_code_exec to execute. Add MBPP
    as a second `task` value the same way.
    """
    from datasets import load_dataset

    src = load_dataset("openai/openai_humaneval", split="test").to_pandas()
    if limit is not None:
        src = src.head(limit)
    rows = []
    for _, r in src.iterrows():
        rows.append(
            {
                "context": r["prompt"],  # function signature + docstring; the model completes this
                "question": "",
                "answer_prefix": "",
                "answer": json.dumps(
                    {"test": r["test"], "entry_point": r["entry_point"], "canonical_solution": r["canonical_solution"]}
                ),
                "task": "humaneval",
                "max_new_tokens": CODE_EXEC_MAX_NEW_TOKENS,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


# --------------------------------------------------------------------------------------------------
# tool_call — pick one tool from a long shared catalog and emit a single JSON call.
# --------------------------------------------------------------------------------------------------

# Each tool: name, human description, a JSON Schema for its arguments, and concrete requests
# (natural-language ask + the gold argument values). The catalog is shared across rows, so a prefill
# press compresses it once and answers every request from the compressed cache — the regime where a
# dropped/garbled tool definition shows up as an invalid call. Swapping in BFCL/xLAM/ToolACE is a
# drop-in: produce the same columns. To lengthen the context, add more tools.
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "units": {"type": "string", "enum": ["c", "f"]}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "requests": [
            {"question": "What's the weather in Paris right now?", "arguments": {"city": "Paris"}},
            {
                "question": "Get the current temperature in Tokyo in Fahrenheit.",
                "arguments": {"city": "Tokyo", "units": "f"},
            },
        ],
    },
    {
        "name": "get_stock_price",
        "description": "Look up the latest price for a stock ticker.",
        "schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
        "requests": [
            {"question": "How much is Apple stock trading at? Its ticker is AAPL.", "arguments": {"ticker": "AAPL"}}
        ],
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "schema": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "subject"],
            "additionalProperties": False,
        },
        "requests": [
            {
                "question": "Email alice@example.com with the subject Hello.",
                "arguments": {"to": "alice@example.com", "subject": "Hello"},
            },
        ],
    },
    {
        "name": "create_calendar_event",
        "description": "Create a calendar event.",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "duration_minutes": {"type": "integer"},
            },
            "required": ["title", "date"],
            "additionalProperties": False,
        },
        "requests": [
            {
                "question": "Put a 'Dentist' appointment on my calendar for 2026-07-01.",
                "arguments": {"title": "Dentist", "date": "2026-07-01"},
            },
        ],
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
            "additionalProperties": False,
        },
        "requests": [
            {
                "question": "Convert 100 USD to EUR.",
                "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
            },
        ],
    },
    {
        "name": "search_flights",
        "description": "Search for flights between two airports on a date.",
        "schema": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}},
            "required": ["origin", "destination", "date"],
            "additionalProperties": False,
        },
        "requests": [
            {
                "question": "Find flights from SFO to JFK on 2026-08-12.",
                "arguments": {"origin": "SFO", "destination": "JFK", "date": "2026-08-12"},
            },
        ],
    },
    {
        "name": "translate_text",
        "description": "Translate text into a target language.",
        "schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}},
            "required": ["text", "target_language"],
            "additionalProperties": False,
        },
        "requests": [
            {
                "question": "Translate 'good morning' into Spanish.",
                "arguments": {"text": "good morning", "target_language": "Spanish"},
            },
        ],
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder with a message at a time.",
        "schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}, "time": {"type": "string"}},
            "required": ["message", "time"],
            "additionalProperties": False,
        },
        "requests": [
            {"question": "Remind me to call mom at 18:00.", "arguments": {"message": "call mom", "time": "18:00"}},
        ],
    },
    # Distractors below are rarely the target; they lengthen the catalog so there is real KV to compress.
    {
        "name": "create_invoice",
        "description": "Create an invoice for a customer.",
        "schema": {
            "type": "object",
            "properties": {
                "customer": {"type": "string"},
                "amount": {"type": "number"},
                "currency": {"type": "string"},
            },
            "required": ["customer", "amount"],
            "additionalProperties": False,
        },
        "requests": [],
    },
    {
        "name": "get_directions",
        "description": "Get driving directions between two places.",
        "schema": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "mode": {"type": "string"}},
            "required": ["origin", "destination"],
            "additionalProperties": False,
        },
        "requests": [],
    },
    {
        "name": "play_music",
        "description": "Play a song or playlist.",
        "schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "shuffle": {"type": "boolean"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "requests": [],
    },
    {
        "name": "book_hotel",
        "description": "Book a hotel room in a city for a date range.",
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "check_in": {"type": "string"}, "check_out": {"type": "string"}},
            "required": ["city", "check_in", "check_out"],
            "additionalProperties": False,
        },
        "requests": [],
    },
]

CATALOG_HEADER = (
    "You are a function-calling assistant with access to the following tools. For the user's request, "
    "respond with ONLY a single JSON object of the form "
    '{"name": <tool name>, "arguments": {<arg>: <value>, ...}} and nothing else.\n\nTools:\n'
)


def render_catalog() -> str:
    lines = [CATALOG_HEADER]
    for i, tool in enumerate(TOOLS, 1):
        lines.append(
            f"{i}. {tool['name']} — {tool['description']}\n"
            f"   parameters (JSON Schema): {json.dumps(tool['schema'])}\n"
        )
    return "".join(lines)


def build_tool_call_df() -> pd.DataFrame:
    """Build the tool_call DataFrame from the inline TOOLS catalog (no network needed)."""
    context = render_catalog()
    rows = []
    for tool in TOOLS:
        for req in tool["requests"]:
            rows.append(
                {
                    "context": context,
                    "question": "\nUser request: " + req["question"] + "\n",
                    "answer_prefix": "",
                    "answer": json.dumps(
                        {"name": tool["name"], "arguments": req["arguments"], "schema": tool["schema"]}
                    ),
                    "task": "tool_call",
                    "max_new_tokens": TOOL_CALL_MAX_NEW_TOKENS,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)
