import pathlib
import unittest


class GraphContractTests(unittest.TestCase):
    def test_supervisor_turn_guards_exist(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8")
        self.assertIn("max_supervisor_turns", content)
        self.assertIn("Supervisor turn limit reached", content)

    def test_graph_uses_repair_and_larger_token_budget(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8")
        self.assertIn("repair_structured_json", content)
        self.assertIn("LLM_MAX_OUTPUT_TOKENS", content)
        self.assertIn("_run_static_analysis", content)
        self.assertIn("_rollback_to_snapshot", content)
        self.assertIn("rollback_snapshot_path", content)

    def test_generation_completion_guards_block_starter_template(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        graph_content = (root / "graph.py").read_text(encoding="utf-8")
        prompt_content = (root / "prompt.py").read_text(encoding="utf-8")

        self.assertIn("_completion_guard_failure", graph_content)
        self.assertIn("_frontend_entry_has_template_boilerplate", graph_content)
        self.assertIn("TemplateBoilerplateError", graph_content)
        self.assertIn("pending implementation steps", graph_content)
        self.assertIn("_frontend_entry_files", graph_content)
        self.assertIn("target_entry", graph_content)
        self.assertIn("/src/main.jsx", prompt_content)
        self.assertIn("Do NOT change it to `/src/index.tsx`", prompt_content)

    def test_vite_entry_detection_uses_rendered_app_file(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8")

        self.assertIn("_FRONTEND_BOOTSTRAP_FILES", content)
        self.assertIn("src/main.jsx", content)
        self.assertIn("imported_paths = re.findall", content)
        self.assertIn("target_entry not in existing_filepaths", content)
        self.assertIn("normalized_path == target_entry", content)

    def test_executor_and_verify_use_deterministic_verification(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8")

        self.assertIn("from runtime_detection import build_run_plan", content)
        self.assertIn("_run_project_verification", content)
        self.assertIn("npm run build", content)
        self.assertIn("npx tsc", content)
        self.assertIn("Deterministic verification", content)
        self.assertNotIn('label="tester"', content)

    def test_graph_suppresses_known_nvidia_kimi_tool_warning(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8")

        self.assertIn("warnings.catch_warnings()", content)
        self.assertIn("moonshotai/kimi-k2", content)
        self.assertIn("not known to support tools", content)
        self.assertIn("type is unknown and inference may fail", content)

    def test_update_coder_prioritizes_named_paths_and_long_requests(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8")

        self.assertIn("_extract_request_paths", content)
        self.assertIn("_UPDATE_REQUEST_CLIP_CHARS", content)
        self.assertIn("Paths explicitly named by the user", content)
        self.assertIn("ImplementationTask", content)
        self.assertIn("Update run detected; preserving exact user request", content)
        self.assertIn('"status": "delegating_update_coder"', content)

    def test_provider_models_are_lazy_at_import(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8")

        self.assertIn("OPENROUTER_MINIMAX_LLM = _LazyModel", content)
        self.assertIn("FIREWORKS_DEEPSEEK_LLM = _LazyModel", content)
        self.assertIn("GROQ_BALANCED_LLM = _LazyModel", content)
        self.assertIn("_resolve_model_candidate(model)", content)

    def test_requirements_and_adapter_have_safe_fallbacks(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8")

        self.assertIn("_fallback_requirements_spec", content)
        self.assertIn("Using deterministic fallback dependencies", content)
        self.assertIn("adapter_failed = str(output).lstrip().startswith", content)
        self.assertIn('status="error" if adapter_failed else "success"', content)

    def test_github_scout_runs_before_template_copy(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8")

        self.assertIn('g.add_conditional_edges("github_search", route_after_github_search', content)
        self.assertIn('g.add_conditional_edges("github_scout", route_after_github_scout', content)
        self.assertIn('g.add_edge("adapter", "repo_finalize")', content)
        self.assertIn('g.add_edge("repo_finalize", "verify")', content)
        self.assertIn('g.add_edge("architect", "requirements")', content)
        self.assertIn('g.add_edge("requirements", "coder")', content)
        self.assertIn('g.add_conditional_edges("coder", route_after_coder', content)
        self.assertIn('g.add_conditional_edges("verify", route_after_verify', content)
        self.assertNotIn('g.add_node("tester", tester_agent)', content)
        self.assertNotIn('g.add_node("reviewer", reviewer_agent)', content)
        self.assertNotIn('g.add_edge("requirements", "github_scout")', content)
        self.assertIn("_scout_downloaded_repo", content)
        self.assertIn("_scout_framework_matches_plan", content)
        self.assertIn("restoring pre-scout project", content)
        self.assertIn("create_project_snapshot", content)
        self.assertIn("skipping {template_name} template copy", content)
        self.assertIn("template_copied = False", content)
        self.assertIn("if is_frontend and template_copied", content)
        self.assertIn("if template_copied:", content)
        self.assertIn("_expected_package_manager", content)
        self.assertIn("Overriding mismatched package manager", content)

    def test_github_search_runs_before_initial_planning(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        graph_content = (root / "graph.py").read_text(encoding="utf-8")
        prompt_content = (root / "prompt.py").read_text(encoding="utf-8")

        self.assertIn('g.add_node("github_search", github_search_agent)', graph_content)
        self.assertIn("route_after_github_search", graph_content)
        self.assertIn('"status": "delegating_github_search"', graph_content)
        self.assertIn("github_search_completed", graph_content)
        self.assertIn("planner_existing_repo_prompt", graph_content)
        self.assertIn("planner_scratch_prompt", graph_content)
        self.assertIn("GitHub search must run before initial planning", graph_content)
        self.assertIn("Prefer adapting a relevant GitHub repo", prompt_content)
        self.assertIn("NEVER list `node_modules`", prompt_content)


if __name__ == "__main__":
    unittest.main()
