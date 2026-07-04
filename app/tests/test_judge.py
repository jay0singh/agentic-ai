import json

import pytest

import core.graph as graph

from helpers import make_state


@pytest.fixture(autouse=True)
def no_langfuse(monkeypatch):
    monkeypatch.setattr(graph, "_langfuse_handler", None)


class TestJudgeNode:
    def test_tool_error_accepts_without_llm_call(self, monkeypatch):
        def fail(prompt):
            raise AssertionError("judge LLM should not be called on tool errors")
        monkeypatch.setattr(graph, "_call_judge", fail)

        state = make_state("q", context=["GitHub API Error: rate limit"], response="answer")
        result = graph.judge_node(state)
        assert result["judge_decision"] == "accept"

    def test_no_context_accepts_without_llm_call(self, monkeypatch):
        def fail(prompt):
            raise AssertionError("judge LLM should not be called without context")
        monkeypatch.setattr(graph, "_call_judge", fail)

        result = graph.judge_node(make_state("hello", response="Hi there!"))
        assert result["judge_decision"] == "accept"

    def test_accept_decision(self, monkeypatch):
        monkeypatch.setattr(graph, "_call_judge", lambda p: json.dumps(
            {"reasoning": "grounded", "decision": "accept"}))
        state = make_state("q", context=["relevant chunk"], response="answer")
        result = graph.judge_node(state)
        assert result["judge_decision"] == "accept"
        assert result["judge_log"] == ["ACCEPT: grounded"]

    def test_retry_decision_carries_rewritten_query(self, monkeypatch):
        monkeypatch.setattr(graph, "_call_judge", lambda p: json.dumps(
            {"reasoning": "off-topic", "decision": "retry", "rewritten_query": "better query"}))
        state = make_state("q", context=["chunk"], response="bad answer")
        result = graph.judge_node(state)
        assert result["judge_decision"] == "retry"
        assert result["rewritten_query"] == "better query"

    def test_unparseable_judge_output_defaults_to_accept(self, monkeypatch):
        monkeypatch.setattr(graph, "_call_judge", lambda p: "not json at all")
        state = make_state("q", context=["chunk"], response="answer")
        result = graph.judge_node(state)
        assert result["judge_decision"] == "accept"


class TestRouteAfterJudge:
    def test_accept_ends(self):
        assert graph.route_after_judge({"judge_decision": "accept", "retry_count": 0}) == "end"

    def test_retry_under_limit_rewrites(self):
        assert graph.route_after_judge({"judge_decision": "retry", "retry_count": 1}) == "rewrite"

    def test_retry_at_limit_goes_to_hitl(self):
        assert graph.route_after_judge({"judge_decision": "retry", "retry_count": 3}) == "hitl"


class TestRewriteNode:
    def test_resets_state_for_retry(self):
        state = make_state("old query", retry_count=0, rewritten_query="new query")
        result = graph.rewrite_node(state)
        assert result["query"] == "new query"
        assert result["retry_count"] == 1
        assert result["context"] == []
        assert result["steps_remaining"] is None
        assert result["next_node"] == "router"

    def test_identical_rewrite_accepts_and_ends(self):
        state = make_state("same query", retry_count=0, rewritten_query="same query")
        result = graph.rewrite_node(state)
        assert result["judge_decision"] == "accept"
        assert "query" not in result
        # Must END, not re-enter the router — regression test for the infinite
        # generate -> judge -> rewrite loop when the judge repeats the query.
        assert result["next_node"] == "end"

    def test_max_retries_accepts_and_ends(self):
        state = make_state("q", retry_count=3, rewritten_query="different")
        result = graph.rewrite_node(state)
        assert result["judge_decision"] == "accept"
        assert result["next_node"] == "end"


class TestHitlNode:
    def test_persists_question_to_queue(self, monkeypatch):
        captured = {}

        def fake_add(question, attempted_answer, judge_reason):
            captured.update(question=question, answer=attempted_answer, reason=judge_reason)
            return 42

        monkeypatch.setattr("core.hitl.add_to_queue", fake_add)
        state = make_state("hard question", response="weak answer",
                           judge_log=["RETRY: not grounded"], retry_count=3)
        result = graph.hitl_node(state)

        assert result["steps_taken"][-1] == "hitl"
        assert captured == {"question": "hard question", "answer": "weak answer",
                            "reason": "RETRY: not grounded"}

    def test_db_failure_does_not_break_response(self, monkeypatch):
        def boom(*args):
            raise RuntimeError("db down")

        monkeypatch.setattr("core.hitl.add_to_queue", boom)
        result = graph.hitl_node(make_state("q", response="a", retry_count=3))
        assert result["steps_taken"][-1] == "hitl"


class TestRouteAfterRewrite:
    def test_accept_and_stop_ends_graph(self):
        assert graph.route_after_rewrite({"next_node": "end"}) == "end"

    def test_genuine_retry_reenters_router(self):
        assert graph.route_after_rewrite({"next_node": "router"}) == "router"

    def test_missing_next_node_defaults_to_router(self):
        assert graph.route_after_rewrite({}) == "router"
