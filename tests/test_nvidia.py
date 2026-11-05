import pathlib
import unittest


class NvidiaChatContractTests(unittest.TestCase):
    def test_server_chatbot_uses_nvidia_model(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        content = (root / "server.py").read_text(encoding="utf-8")

        self.assertIn("from langchain_nvidia_ai_endpoints import ChatNVIDIA", content)
        self.assertIn("def _make_chat_llm", content)
        self.assertIn("_chat_llm_instance = None", content)
        self.assertIn("def get_chat_llm", content)
        self.assertIn("_chat_llm_instance = _make_chat_llm(_nvidia_chat_kwargs)", content)
        self.assertIn("asyncio.to_thread(model.invoke, messages)", content)
        self.assertIn("def _bind_chat_tools", content)
        self.assertIn("warnings.filterwarnings", content)
        self.assertIn("type is unknown and inference may fail", content)
        self.assertIn("_bind_chat_tools(get_chat_llm(), chat_tools_list)", content)
        self.assertIn('"NVIDIA_API_KEY"', content)
        self.assertIn('"NVIDIA_CHAT_MODEL"', content)
        self.assertIn('"moonshotai/kimi-k2.6"', content)
        self.assertNotIn("CHAT_LLM = ChatOpenAI", content)
        self.assertNotIn(".ainvoke(current_messages)", content)
        self.assertNotIn(".astream(current_messages)", content)
        self.assertNotIn("models.inference.ai.azure.com", content)
        self.assertNotIn("ghp_", content)
        self.assertNotIn("nvapi-", content)


if __name__ == "__main__":
    unittest.main()
