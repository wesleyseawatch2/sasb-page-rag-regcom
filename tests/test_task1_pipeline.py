import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "task1"))

from task1_pipeline import (  # noqa: E402
    hit_at,
    parse_jsonish,
    reciprocal_rank,
    reciprocal_rank_fusion,
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

    def test_retrieval_metrics(self):
        self.assertEqual(hit_at([2, 7, 9], {7}, 1), 0.0)
        self.assertEqual(hit_at([2, 7, 9], {7}, 5), 1.0)
        self.assertEqual(reciprocal_rank([2, 7, 9], {7}), 0.5)

    def test_json_fence_is_accepted(self):
        self.assertEqual(parse_jsonish('```json\n{"no_relevant_page": true}\n```')["no_relevant_page"], True)


if __name__ == "__main__":
    unittest.main()
