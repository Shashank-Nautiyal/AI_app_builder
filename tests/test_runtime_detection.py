import json
import pathlib
import tempfile
import unittest

from codemind.core import runtime_detection


class RuntimeDetectionTests(unittest.TestCase):
    def test_detect_python_app_file_prefers_framework_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            (root / "backend").mkdir()
            (root / "backend" / "main.py").write_text("print('hello')\n", encoding="utf-8")
            (root / "backend" / "server.py").write_text(
                "from fastapi import FastAPI\napp = FastAPI()\nif __name__ == '__main__':\n    pass\n",
                encoding="utf-8",
            )

            detected = runtime_detection.detect_python_app_file(root / "backend")
            self.assertEqual(detected, "server.py")

    def test_detect_node_start_command_falls_back_to_main_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            (root / "src").mkdir(parents=True)
            (root / "package.json").write_text(
                json.dumps({"name": "api", "main": "src/api.js"}, indent=2),
                encoding="utf-8",
            )
            (root / "src" / "api.js").write_text(
                "import express from 'express';\nconst app = express();\napp.listen(3000);\n",
                encoding="utf-8",
            )

            command = runtime_detection.detect_node_start_command(root)
            self.assertEqual(command, "npm install && node src/api.js")

    def test_detect_frontend_dev_script_uses_npm_run_dev(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            (root / "package.json").write_text(
                json.dumps(
                    {
                        "name": "frontend",
                        "scripts": {"dev": "vite", "build": "vite build"},
                        "dependencies": {"react": "^18.3.1", "react-dom": "^18.3.1"},
                        "devDependencies": {"vite": "^5.0.8"},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            command = runtime_detection.detect_node_start_command(root)
            plan = runtime_detection.build_run_plan(root)

            self.assertEqual(command, "npm install && npm run dev")
            self.assertEqual(plan["serve_port"], 5173)

    def test_detect_node_start_command_skips_install_when_node_modules_exists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            (root / "node_modules").mkdir()
            (root / "package.json").write_text(
                json.dumps({"name": "api", "scripts": {"start": "node server.js"}}, indent=2),
                encoding="utf-8",
            )

            command = runtime_detection.detect_node_start_command(root)
            self.assertEqual(command, "npm start")

    def test_build_run_plan_discovers_nested_frontend_and_backend(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            (root / "backend").mkdir()
            (root / "frontend").mkdir()

            (root / "backend" / "package.json").write_text(
                json.dumps(
                    {
                        "name": "backend",
                        "scripts": {"start": "node src/server.js"},
                        "main": "src/server.js",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (root / "backend" / "src").mkdir()
            (root / "backend" / "src" / "server.js").write_text(
                "import express from 'express';\nconst app = express();\napp.listen(3000);\n",
                encoding="utf-8",
            )

            (root / "frontend" / "package.json").write_text(
                json.dumps(
                    {
                        "name": "frontend",
                        "scripts": {"start": "react-scripts start"},
                        "dependencies": {"react": "^18.3.1", "react-dom": "^18.3.1"},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            plan = runtime_detection.build_run_plan(root)
            commands = {command["cmd"] for command in plan["commands"]}

            self.assertIn('cd /d "backend" && npm install && npm start\r\n', commands)
            self.assertIn('cd /d "frontend" && npm install && npm start\r\n', commands)
            self.assertEqual(plan["serve_port"], 3000)


if __name__ == "__main__":
    unittest.main()
