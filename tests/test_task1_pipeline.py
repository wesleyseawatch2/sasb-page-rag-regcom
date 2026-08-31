import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "task1"))

from task1_pipeline import (  # noqa: E402
    build_candidate_pool,
    hit_at,
    parse_jsonish,
    reciprocal_rank,
    reciprocal_rank_fusion,
    normalize_vlm_ranking,
    source_pdf_stem,
)


class Task1PipelineTests(unittest.TestCase):
    def test_source_pdf_stem_removes_only_page_suffix(self):
        self.assertEqual(source_pdf_stem("report_name_012"), "report_name")
        self.assertEqual(source_pdf_stem("report_name_final"), "report_name_final")

    def test_rrf_combines_independent_rankings(self):
        rankings = {
            "lexical": [{"page": 2, "rank": 1}, {"page": 1, "rank": 2}],
            "rules": [{"page": 1, "rank": 1}],
        }
        fused = reciprocal_rank_fusion(rankings, {"lexical": 1.0, "rules": 1.0}, 60)
        self.assertEqual(fused[0]["page"], 1)
        self.assertEqual(fused[0]["lane_ranks"], {"lexical": 2, "rules": 1})

    def test_candidate_pool_is_lane_union(self):
        rankings = {
            "word": [{"page": 1, "rank": 1}, {"page": 2, "rank": 2}],
            "char": [{"page": 3, "rank": 1}, {"page": 2, "rank": 2}],
        }
        fused = [{"page": 1, "score": 0.1}, {"page": 2, "score": 0.05}, {"page": 3, "score": 0.01}]
        # The pool must retain pages contributed by either lane.
        pool = build_candidate_pool(rankings, fused, 2)
        self.assertEqual({item["page"] for item in pool}, {1, 2, 3})

    def test_retrieval_metrics(self):
        self.assertEqual(hit_at([2, 7, 9], {7}, 1), 0.0)
        self.assertEqual(hit_at([2, 7, 9], {7}, 5), 1.0)
        self.assertEqual(reciprocal_rank([2, 7, 9], {7}), 0.5)

    def test_json_fence_is_accepted(self):
        self.assertEqual(parse_jsonish('```json\n{"no_relevant_page": true}\n```')["no_relevant_page"], True)

    def test_vlm_ranking_maps_candidate_keys_and_removes_hallucinated_pages(self):
        candidates = [{"page": 12}, {"page": 27}]
        ranked = [{"candidate_key": "C02", "relevance": "0.8"}, {"page": 999}]
        self.assertEqual([r["page"] for r in normalize_vlm_ranking(ranked, candidates)], [27, 12])


if __name__ == "__main__":
    unittest.main()
