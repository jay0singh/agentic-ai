def make_state(query, **overrides):
    """A complete AgentState dict with sensible defaults for node tests."""
    state = {
        "query": query,
        "original_query": query,
        "history": "",
        "top_k": 3,
        "context": [],
        "citations": [],
        "steps_taken": [],
        "steps_remaining": None,
        "next_node": "router",
        "parameters": {},
        "response": None,
        "retry_count": 0,
        "judge_decision": "accept",
        "rewritten_query": None,
        "judge_log": [],
    }
    state.update(overrides)
    return state


class FakeModel:
    """Stands in for the Groq chat model; returns a canned .content."""
    def __init__(self, content="", error=None):
        self.content = content
        self.error = error

    def invoke(self, messages):
        if self.error:
            raise self.error
        return type("Msg", (), {"content": self.content})()
