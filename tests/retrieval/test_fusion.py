import pytest

from repo_recall.retrieval.fusion import reciprocal_rank_fusion
from repo_recall.retrieval.models import SearchMode


def test_chunk_in_both_lists_beats_chunks_in_one(ranked_list):
    fused = reciprocal_rank_fusion(
        {
            "vector": ranked_list(["a", "shared"]),
            "keyword": ranked_list(["b", "shared"], SearchMode.KEYWORD),
        },
        k=3,
    )
    assert fused[0].chunk_id == "shared"
    assert {r.chunk_id for r in fused[1:]} == {"a", "b"}


def test_rrf_score_is_sum_of_reciprocal_ranks(ranked_list):
    fused = reciprocal_rank_fusion(
        {"vector": ranked_list(["x", "a"]), "keyword": ranked_list(["b", "c", "a"])},
        k=5,
        rrf_k=60,
    )
    a = next(r for r in fused if r.chunk_id == "a")
    assert a.score == pytest.approx(1 / 62 + 1 / 63)
    assert a.component_ranks == {"vector": 2, "keyword": 3}


def test_fused_results_are_reranked_hybrid_and_capped_at_k(ranked_list):
    fused = reciprocal_rank_fusion(
        {"vector": ranked_list(list("abcdef")), "keyword": ranked_list(list("ghijkl"))}, k=4
    )
    assert len(fused) == 4
    assert [r.rank for r in fused] == [1, 2, 3, 4]
    assert all(r.mode is SearchMode.HYBRID for r in fused)
    assert fused[0].score >= fused[-1].score


def test_single_list_component_ranks_only_name_that_list(ranked_list):
    fused = reciprocal_rank_fusion({"vector": ranked_list(["a"]), "keyword": []}, k=5)
    assert fused[0].component_ranks == {"vector": 1}


def test_empty_lists_give_no_results():
    assert reciprocal_rank_fusion({"vector": [], "keyword": []}, k=5) == []
