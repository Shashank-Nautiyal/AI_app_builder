import unittest
from unittest.mock import patch

from codemind.services import github_scout


class GitHubScoutSearchTests(unittest.TestCase):
    def test_build_queries_keeps_exact_prompt_first_then_llm_phrases(self):
        queries = github_scout._build_github_search_queries(
            "create ai based car market place",
            "React Vite",
            max_queries=3,
            llm_phrases=["vehicle marketplace", "automotive ecommerce platform"],
        )

        self.assertEqual(len(queries), 3)
        self.assertTrue(queries[0].startswith("ai based car market place react "))
        self.assertIn("in:readme,description", queries[0])
        self.assertIn("language:JavaScript", queries[0])
        self.assertTrue(queries[1].startswith("vehicle marketplace "))
        self.assertTrue(queries[2].startswith("automotive ecommerce platform "))

    def test_llm_phrase_parser_accepts_json_shapes_and_sanitizes(self):
        phrases = github_scout._parse_llm_phrase_list(
            '{"queries": ["Vehicle Marketplace", "used car dealer platform", "AI car sales app"]}'
        )

        self.assertEqual(
            phrases,
            ["vehicle marketplace", "used car dealer platform", "ai car sales app"],
        )

    def test_search_uses_multiple_queries_and_dedupes_repos(self):
        def fake_gh_get(_url, params):
            query = params["q"]
            if query.startswith("vehicle marketplace"):
                return {"items": [{"id": 2, "full_name": "demo/vehicles"}]}
            return {
                "items": [
                    {"id": 1, "full_name": "demo/exact"},
                    {"id": 2, "full_name": "demo/vehicles"},
                ]
            }

        with patch.object(
            github_scout,
            "_build_github_search_queries",
            return_value=[
                "ai car marketplace in:readme,description language:JavaScript",
                "vehicle marketplace in:readme,description language:JavaScript",
            ],
        ), patch.object(github_scout, "_gh_get", side_effect=fake_gh_get):
            repos = github_scout.search_github_repos("create ai car marketplace", "React", max_results=5)

        self.assertEqual([repo["id"] for repo in repos], [1, 2])

    def test_attendance_face_prompt_gets_domain_fallback_phrases(self):
        queries = github_scout._build_github_search_queries(
            "create a attandance system that uses face for mark the attendance",
            "React Vite Node Express",
            max_queries=4,
            llm_phrases=[],
        )

        joined = "\n".join(queries)
        self.assertIn("attendance management system", joined)
        self.assertIn("facial recognition attendance", joined)

    def test_rank_requires_relevance_before_star_count(self):
        repos = [
            {
                "full_name": "public-apis/public-apis",
                "description": "A collective list of free APIs",
                "stargazers_count": 435000,
                "pushed_at": "2026-05-15T15:50:51Z",
                "fork": False,
                "archived": False,
            },
            {
                "full_name": "demo/face-attendance",
                "description": "Face recognition attendance system for employee check ins",
                "stargazers_count": 1200,
                "pushed_at": "2026-05-01T10:00:00Z",
                "fork": False,
                "archived": False,
            },
        ]

        selected = github_scout.rank_and_select_repo(repos, "create a face detection attendance system")

        self.assertEqual(selected["full_name"], "demo/face-attendance")


if __name__ == "__main__":
    unittest.main()
