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


class TestVectorSearchNode:
    def test_uses_top_k_and_collects_citations(self, monkeypatch):
        calls = {}

        def fake_retrieve(query, top_k=3):
            calls["top_k"] = top_k
            return [
                {"content": "chunk one", "source": "handbook.docx", "distance": 0.21},
                {"content": "chunk two", "source": "handbook.docx", "distance": 0.25},
            ]

        monkeypatch.setattr(graph, "retrieve", fake_retrieve)
        state = make_state("q", top_k=7, parameters={"search_query": "return policy"})
        result = graph.vector_search_node(state)

        assert calls["top_k"] == 7
        assert result["citations"] == [
            {"source": "handbook.docx", "distance": 0.21},
            {"source": "handbook.docx", "distance": 0.25},
        ]
        assert "[source: handbook.docx]" in result["context"][0]
        assert result["next_node"] == "router"

    def test_keyword_only_match_has_null_distance_citation(self, monkeypatch):
        monkeypatch.setattr(graph, "retrieve", lambda query, top_k=3: [
            {"content": "chunk", "source": "handbook.docx", "distance": None},
        ])
        state = make_state("q", parameters={"search_query": "section 7.4"})
        result = graph.vector_search_node(state)
        assert result["citations"] == [{"source": "handbook.docx", "distance": None}]

    def test_no_relevant_chunks_leaves_state_unchanged(self, monkeypatch):
        monkeypatch.setattr(graph, "retrieve", lambda query, top_k=3: [])
        state = make_state("q", parameters={"search_query": "x"})
        result = graph.vector_search_node(state)
        assert result["context"] == []
        assert result["citations"] == []


class TestGeneratorNode:
    def test_oversized_context_is_capped(self, monkeypatch):
        captured = {}

        class CapturingModel:
            def invoke(self, messages):
                captured["prompt"] = messages[0].content
                return type("Msg", (), {"content": "ok"})()

        monkeypatch.setattr(graph, "chat_model", CapturingModel())
        # Simulate top_k=10 + long web results: way beyond any token budget
        state = make_state("q", context=["x" * 50_000], steps_taken=["vector_search"])
        result = graph.generator_node(state)

        assert result["next_node"] == "judge"
        assert "[context truncated]" in captured["prompt"]
        # Prompt = capped context + instructions; must stay near the cap
        assert len(captured["prompt"]) < graph.MAX_GENERATOR_CONTEXT_CHARS + 3_000


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
