import core.graph as graph

from helpers import make_state, FakeModel


class TestRouterNode:
    def test_casual_query_goes_straight_to_generator(self):
        result = graph.router_node(make_state("hello"))
        assert result["next_node"] == "generate"
        assert result["steps_remaining"] == []

    def test_nimbuscart_keyword_forces_vector_search(self):
        result = graph.router_node(make_state("Tell me about NimbusDirect"))
        assert result["next_node"] == "vector_search"
        assert result["parameters"]["search_query"] == "Tell me about NimbusDirect"

    def test_classifier_plan_is_executed_in_order(self, monkeypatch):
        monkeypatch.setattr(graph, "classify_query", lambda q: {
            "tools": ["web_search"],
            "search_query": "FIFA World Cup 2022 winner",
            "github_parameters": {},
        })
        result = graph.router_node(make_state("Who won the last World Cup?"))
        assert result["next_node"] == "web_search"
        assert result["steps_remaining"] == []
        assert result["parameters"]["search_query"] == "FIFA World Cup 2022 winner"

    def test_multiple_tools_pop_sequentially(self):
        state = make_state(
            "irrelevant",
            steps_remaining=["vector_search", "web_search"],
            parameters={"search_query": "x", "github": {}},
        )
        result = graph.router_node(state)
        assert result["next_node"] == "vector_search"
        assert result["steps_remaining"] == ["web_search"]

    def test_all_tools_done_routes_to_generator(self):
        state = make_state("irrelevant", steps_remaining=[])
        result = graph.router_node(state)
        assert result["next_node"] == "generate"

    def test_no_tools_falls_back_to_vector_search(self, monkeypatch):
        monkeypatch.setattr(graph, "classify_query", lambda q: {
            "tools": [], "search_query": "", "github_parameters": {},
        })
        result = graph.router_node(make_state("Something substantive here"))
        assert result["next_node"] == "vector_search"
        assert result["parameters"]["search_query"] == "Something substantive here"


class TestResolveFollowup:
    def test_rewrites_followup(self, monkeypatch):
        monkeypatch.setattr(
            graph, "chat_model",
            FakeModel(content='"Does returning an item to NimbusCart cost anything?"'),
        )
        resolved = graph.resolve_followup(
            "Does it cost anything?",
            "User: What is the return policy?\nAssistant: 30 days...",
        )
        assert resolved == "Does returning an item to NimbusCart cost anything?"

    def test_falls_back_to_original_on_error(self, monkeypatch):
        monkeypatch.setattr(graph, "chat_model", FakeModel(error=RuntimeError("boom")))
        assert graph.resolve_followup("Does it cost anything?", "history") == "Does it cost anything?"

    def test_falls_back_to_original_on_empty_output(self, monkeypatch):
        monkeypatch.setattr(graph, "chat_model", FakeModel(content="   "))
        assert graph.resolve_followup("Does it cost anything?", "history") == "Does it cost anything?"
