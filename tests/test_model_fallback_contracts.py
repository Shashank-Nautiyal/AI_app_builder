import pathlib
import re
import unittest


class ModelFallbackContractTests(unittest.TestCase):
    def test_graph_has_role_based_limit_fallback_routes(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8")

        self.assertIn("_LIMIT_FALLBACK_ROUTES", content)
        self.assertIn("CLARIFIER_LLM = GROQ_FAST_LLM", content)
        self.assertIn("ARCHITECT_LLM = CEREBRAS_GPT_OSS_LLM", content)
        self.assertIn("INTEGRATOR_LLM = NVIDIA_KIMI_LLM", content)
        self.assertIn('"planner": [GROQ_BALANCED_LLM, MISTRAL_MEDIUM_LLM, CEREBRAS_GPT_OSS_LLM, OPENROUTER_MINIMAX_LLM', content)
        self.assertIn('"architect": [CEREBRAS_GPT_OSS_LLM, MISTRAL_DEVSTRAL_LLM, GROQ_REASONING_LLM, NVIDIA_QWEN_CODER_LLM', content)
        self.assertIn('"coder": [FIREWORKS_DEEPSEEK_LLM, MISTRAL_DEVSTRAL_LLM, NVIDIA_QWEN_CODER_LLM', content)
        self.assertIn('"integrator": [NVIDIA_KIMI_LLM, MISTRAL_MEDIUM_LLM, CEREBRAS_GPT_OSS_LLM', content)
        self.assertIn("OPENROUTER_QWEN_CODER_FREE_LLM", content)
        self.assertIn("_resolve_model_candidate(candidate)", content)
        self.assertIn("isinstance(model, _LazyModel)", content)
        self.assertIn("auto_model_fallback_on_limit", content)
        self.assertIn("_run_agent_with_model_fallback", content)

    def test_fireworks_is_reserved_for_coding_routes(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8")
        routes = dict(re.findall(r'"([^"]+)": \[([^\]]+)\]', content))

        for route_name in ("planner", "architect", "integrator"):
            self.assertEqual(routes[route_name].count("FIREWORKS_"), 0)
        for route_name in ("coder", "update_coder"):
            self.assertLessEqual(routes[route_name].count("FIREWORKS_"), 1)
        self.assertIn("FIREWORKS_DEEPSEEK_LLM", routes["coder"])

    def test_graph_treats_credit_exhaustion_as_limit_error(self):
        graph_source = pathlib.Path(__file__).resolve().parents[1] / "graph.py"
        content = graph_source.read_text(encoding="utf-8").lower()

        self.assertIn("payment required", content)
        self.assertIn("monthly included credits", content)
        self.assertIn("pre-paid credits", content)
        self.assertIn("readerror", content)
        self.assertIn("bad record mac", content)

    def test_server_and_ui_expose_auto_fallback_toggle(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        server_content = (root / "server.py").read_text(encoding="utf-8")
        api_content = (root / "frontend" / "src" / "services" / "api.js").read_text(encoding="utf-8")
        sidebar_content = (root / "frontend" / "src" / "components" / "ProjectSidebar.jsx").read_text(encoding="utf-8")

        self.assertIn("auto_model_fallback_on_limit", server_content)
        self.assertIn("AUTO_MODEL_FALLBACK_STORAGE_KEY", api_content)
        self.assertIn("getAutoModelFallbackSetting", api_content)
        self.assertIn("Auto-switch on model limits", sidebar_content)


if __name__ == "__main__":
    unittest.main()
