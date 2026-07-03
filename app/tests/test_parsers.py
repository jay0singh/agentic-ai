import pytest

from core.graph import parse_json_response, is_casual_query


class TestParseJsonResponse:
    def test_plain_json(self):
        assert parse_json_response('{"decision": "accept"}') == {"decision": "accept"}

    def test_json_embedded_in_text(self):
        raw = 'Here is my answer:\n```json\n{"tools": ["web_search"]}\n```\nDone.'
        assert parse_json_response(raw) == {"tools": ["web_search"]}

    def test_nested_json(self):
        raw = '{"github_parameters": {"owner": "langchain-ai", "repo": "langgraph"}}'
        assert parse_json_response(raw)["github_parameters"]["repo"] == "langgraph"

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            parse_json_response("I could not produce JSON, sorry.")


class TestIsCasualQuery:
    @pytest.mark.parametrize("query", [
        "hello", "Hey", "hi!", "thanks", "good morning",
        "heyy",        # repeated letters collapse
        "hellooo",
        "Hey bot",     # greeting + short tail
    ])
    def test_casual(self, query):
        assert is_casual_query(query) is True

    @pytest.mark.parametrize("query", [
        "What is NimbusCart's return policy?",
        "Who won the FIFA World Cup in 2022?",
        "Read the README for langchain-ai/langgraph",
        "How many vacation days do employees get?",
    ])
    def test_not_casual(self, query):
        assert is_casual_query(query) is False
