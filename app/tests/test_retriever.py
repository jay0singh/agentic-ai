from core.retriever import _rrf_merge


def vhit(content, source="doc.docx", distance=0.2):
    return {"content": content, "source": source, "distance": distance}


def khit(content, source="doc.docx"):
    return {"content": content, "source": source, "distance": None}


def test_chunk_found_by_both_searches_ranks_first():
    vector = [vhit("A"), vhit("B")]
    keyword = [khit("C"), khit("A")]
    merged = _rrf_merge(vector, keyword, top_k=3)
    assert merged[0]["content"] == "A"          # appears in both lists


def test_keyword_only_hit_is_included():
    merged = _rrf_merge([vhit("A")], [khit("K")], top_k=3)
    contents = [m["content"] for m in merged]
    assert "K" in contents


def test_keyword_only_hit_keeps_none_distance():
    merged = _rrf_merge([], [khit("K")], top_k=3)
    assert merged == [{"content": "K", "source": "doc.docx", "distance": None}]


def test_overlapping_hit_keeps_vector_distance():
    merged = _rrf_merge([vhit("A", distance=0.21)], [khit("A")], top_k=3)
    assert merged[0]["distance"] == 0.21


def test_top_k_limits_results():
    vector = [vhit(f"V{i}") for i in range(5)]
    keyword = [khit(f"K{i}") for i in range(5)]
    assert len(_rrf_merge(vector, keyword, top_k=3)) == 3


def test_empty_inputs():
    assert _rrf_merge([], [], top_k=3) == []
