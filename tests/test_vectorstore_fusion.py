"""
Unit tests for HybridStore._fuse's Reciprocal Rank Fusion math - the exact
kind of thing that breaks silently (an off-by-one in rank, a wrong sum
order) without necessarily changing which answer wins in the golden set.

_fuse doesn't touch `self` (no self._collection / self._bm25 access), so
it's called directly on the class with a dummy `self` rather than
constructing a full HybridStore (which would need a live Chroma client).
"""
from vectorstore import HybridStore


def _fuse(dense, sparse, k=60):
    return HybridStore._fuse(None, dense, sparse, k=k)


def _row(cid: str, doc: str = "doc") -> tuple[str, str, dict]:
    return (cid, doc, {"title": cid})


def test_rrf_score_formula_matches_paper_definition():
    dense = [_row("a"), _row("b")]
    sparse = [_row("b"), _row("a")]
    results = {r.chunk_id: r for r in _fuse(dense, sparse, k=60)}
    # a: dense rank 0 -> 1/61, sparse rank 1 -> 1/62
    assert results["a"].rrf_score == 1 / 61 + 1 / 62
    # b: dense rank 1 -> 1/62, sparse rank 0 -> 1/61 (same total, order-independent)
    assert results["b"].rrf_score == 1 / 62 + 1 / 61


def test_chunk_appearing_in_both_lists_outranks_single_list_hit():
    dense = [_row("only_dense"), _row("both")]
    sparse = [_row("both"), _row("only_sparse")]
    fused = _fuse(dense, sparse)
    ids_in_order = [r.chunk_id for r in fused]
    assert ids_in_order[0] == "both"


def test_moderate_hit_in_both_lists_can_outrank_a_single_top_hit():
    # The point of RRF over "just use the dense ranking": a chunk that's
    # merely decent in *both* lists can still beat a chunk that's #1 in
    # only one - being retrieved by both signals is itself evidence.
    dense = [_row("dense_only_top"), _row("shared")]
    sparse = [_row("other1"), _row("other2"), _row("shared")]
    fused = _fuse(dense, sparse)
    assert fused[0].chunk_id == "shared"


def test_single_list_top_hit_beats_same_list_lower_rank():
    # Within a single list, RRF still preserves the original ordering.
    dense = [_row("top"), _row("second")]
    fused = _fuse(dense, [])
    assert [r.chunk_id for r in fused] == ["top", "second"]


def test_dense_and_sparse_ranks_recorded_on_the_result():
    dense = [_row("a"), _row("b")]
    sparse = [_row("b")]
    results = {r.chunk_id: r for r in _fuse(dense, sparse)}
    assert results["a"].dense_rank == 0
    assert results["a"].sparse_rank is None
    assert results["b"].dense_rank == 1
    assert results["b"].sparse_rank == 0


def test_empty_inputs_produce_no_results():
    assert _fuse([], []) == []


def test_sorted_descending_by_score():
    dense = [_row("first"), _row("second"), _row("third")]
    fused = _fuse(dense, [])
    scores = [r.rrf_score for r in fused]
    assert scores == sorted(scores, reverse=True)
