import unittest

from codemind.pipeline import prompt


class PromptContractTests(unittest.TestCase):
    def test_coder_prompt_enforces_frontend_consistency(self):
        content = prompt.coder_system_prompt("")
        self.assertIn("FRONTEND CONSISTENCY", content)
        self.assertIn("prefer Vite", content)
        self.assertIn("react-scripts", content)
        self.assertIn("node src/App.js", content)
        self.assertIn("RUNNABLE PACKAGE.JSON", content)

    def test_executor_prompt_forbids_running_frontend_entry_files_with_node(self):
        content = prompt.executor_system_prompt()
        self.assertIn("FRONTEND TOOLCHAIN RULE", content)
        self.assertIn("npm run dev", content)
        self.assertIn("src/App.js", content)

    def test_requirements_prompt_discourages_mixed_frontend_toolchains(self):
        content = prompt.requirements_prompt("Demo", "React, Vite", "frontend/package.json")
        self.assertIn("ONE coherent toolchain family", content)
        self.assertIn("default to Vite + React", content)
        self.assertIn("Do NOT choose Create React App / react-scripts", content)
        self.assertIn("@types/*", content)

    def test_clarifier_and_planner_prefer_vite_for_new_react_apps(self):
        clarifier = prompt.clarifier_prompt("build a music app")
        planner = prompt.planner_prompt("build a music app")

        self.assertIn("React with Vite", clarifier)
        self.assertIn("Do not choose Create React App / react-scripts", clarifier)
        self.assertIn("prefer Vite over Create React App / react-scripts", planner)

    def test_camera_media_workflows_require_real_capture(self):
        planner = prompt.planner_prompt("create a face detection attendance system")
        architect = prompt.architect_prompt("{}")
        coder = prompt.coder_system_prompt("")

        self.assertIn("navigator.mediaDevices.getUserMedia", planner)
        self.assertIn("real browser camera/upload component", architect)
        self.assertIn("Do not ship only static cards", coder)

    def test_supervisor_prompt_prefers_github_and_verify(self):
        content = prompt.supervisor_prompt("{}")

        self.assertIn("Prefer adapting a relevant GitHub repo", content)
        self.assertIn("- verify:", content)
        self.assertNotIn("- tester:", content)
        self.assertNotIn("- reviewer:", content)


if __name__ == "__main__":
    unittest.main()
