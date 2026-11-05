from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PipelineHardeningContractTests(unittest.TestCase):
    def test_graph_has_manifest_placeholder_and_import_gates(self):
        graph_content = (ROOT / "graph.py").read_text(encoding="utf-8")
        schemas_content = (ROOT / "schemas.py").read_text(encoding="utf-8")

        self.assertIn("GeneratedFileInfo", schemas_content)
        self.assertIn("generated_file_manifest", schemas_content)
        self.assertIn("build_generated_file_manifest", graph_content)
        self.assertIn("placeholder_failure_result(scan_placeholders", graph_content)
        self.assertIn("validate_project_imports(root)", graph_content)
        self.assertIn("format_generated_manifest", graph_content)

    def test_graph_has_runtime_status_resume_and_parallel_hooks(self):
        graph_content = (ROOT / "graph.py").read_text(encoding="utf-8")
        server_content = (ROOT / "server.py").read_text(encoding="utf-8")

        self.assertIn("run_runtime_health_check", graph_content)
        self.assertIn("def _set_project_status", graph_content)
        self.assertIn("def resume_pipeline", graph_content)
        self.assertIn("GRAPH_PARALLEL_CODER", graph_content)
        self.assertIn("parallel_coder_agent", graph_content)
        self.assertIn("PREWARM_LLM_MODELS", server_content)
        self.assertIn("Missing generated-project env values", server_content)

    def test_tools_expose_import_validation_and_async_safe_visual_verify(self):
        tools_content = (ROOT / "Tools.py").read_text(encoding="utf-8")

        self.assertIn("def validate_imports", tools_content)
        self.assertIn("validate_project_imports", tools_content)
        self.assertIn("ThreadPoolExecutor", tools_content)
        self.assertIn("_visual_verify_sync", tools_content)


if __name__ == "__main__":
    unittest.main()
