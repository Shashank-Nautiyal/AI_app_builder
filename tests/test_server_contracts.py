import pathlib
import unittest


class ServerContractTests(unittest.TestCase):
    def test_server_has_error_handlers_and_success_envelope(self):
        server_source = pathlib.Path(__file__).resolve().parents[1] / "server.py"
        content = server_source.read_text(encoding="utf-8")
        self.assertIn("def api_success(", content)
        self.assertIn("@app.exception_handler(HTTPException)", content)
        self.assertIn("@app.exception_handler(Exception)", content)

    def test_server_uses_run_plan_and_final_refresh_delay(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        server_content = (root / "server.py").read_text(encoding="utf-8")
        app_shell_content = (root / "frontend" / "src" / "components" / "AppShell.jsx").read_text(encoding="utf-8")
        sidebar_content = (root / "frontend" / "src" / "components" / "ProjectSidebar.jsx").read_text(encoding="utf-8")

        self.assertIn("build_run_plan(project_dir)", server_content)
        self.assertIn("FINAL_STATUS_EVENT_DELAY_SECONDS", server_content)
        self.assertIn("/api/projects/upload", server_content)
        self.assertIn("relative_paths", server_content)
        self.assertIn('"run_plan": run_plan', server_content)
        self.assertIn('"structure": structure', server_content)
        self.assertIn("build_ui_event", server_content)
        self.assertIn("waitForSettledStatus", app_shell_content)
        self.assertIn("uploadProjectFiles", app_shell_content)
        self.assertIn("webkitdirectory", sidebar_content)


if __name__ == "__main__":
    unittest.main()
