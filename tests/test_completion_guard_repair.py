import json
import pathlib
import tempfile
import unittest

from codemind.pipeline import graph
from codemind.core.schemas import CoderState, File, ImplementationTask, Plan, TaskPlan


class CompletionGuardRepairTests(unittest.TestCase):
    def test_template_boilerplate_is_repaired_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            src = root / "src"
            src.mkdir()

            (root / "package.json").write_text(
                json.dumps(
                    {
                        "name": "react-vite",
                        "type": "module",
                        "scripts": {"build": "vite build"},
                    }
                ),
                encoding="utf-8",
            )
            (root / "vite.config.js").write_text("export default {}\n", encoding="utf-8")
            (root / "index.html").write_text(
                "<html><head><title>react-vite</title></head><body><div id=\"root\"></div></body></html>\n",
                encoding="utf-8",
            )
            (src / "main.jsx").write_text("import App from './App.jsx'\n", encoding="utf-8")
            (src / "App.jsx").write_text(
                "\n".join(
                    [
                        "import reactLogo from './assets/react.svg'",
                        "import viteLogo from './assets/vite.svg'",
                        "export default function App() {",
                        "  return <h1>Get started</h1>",
                        "}",
                        "const text = 'Count is 0 and HMR is ready'",
                    ]
                ),
                encoding="utf-8",
            )

            task_plan = TaskPlan(
                implementation_steps=[
                    ImplementationTask(
                        filepath="src/App.jsx",
                        action="rewrite",
                        task_description="Rewrite the frontend entry file for the requested application.",
                        dependencies=[],
                    )
                ]
            )
            state = {
                "plan": Plan(
                    name="Face Detection Attendance System",
                    description="A face detection attendance system for employee check-ins.",
                    techstack="React, Vite",
                    project_type="fullstack",
                    features=["Face Detection", "Attendance Tracking", "User Authentication"],
                    files=[File(path="src/App.jsx", purpose="Main React component")],
                ),
                "task_plan": task_plan,
                "coder_state": CoderState(task_plan=task_plan, current_step_idx=1),
            }
            config = {"configurable": {"project_root": str(root)}}

            failure = graph._completion_guard_failure(state, config)

            self.assertIsNone(failure)
            app_content = (src / "App.jsx").read_text(encoding="utf-8")
            self.assertIn("Face Detection Attendance System", app_content)
            self.assertIn("Attendance Tracking", app_content)
            self.assertNotIn("Get started", app_content)
            self.assertNotIn("reactLogo", app_content)
            self.assertTrue((src / "App.css").exists())
            self.assertIn(
                "<title>Face Detection Attendance System</title>",
                (root / "index.html").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                json.loads((root / "package.json").read_text(encoding="utf-8"))["name"],
                "face-detection-attendance-system",
            )

    def test_face_attendance_without_capture_path_fails_completion_guard(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            src = root / "src"
            src.mkdir()

            (root / "package.json").write_text(
                json.dumps({"name": "face-attendance", "type": "module"}),
                encoding="utf-8",
            )
            (root / "vite.config.js").write_text("export default {}\n", encoding="utf-8")
            (src / "main.jsx").write_text("import App from './App.jsx'\n", encoding="utf-8")
            (src / "App.jsx").write_text(
                "export default function App() { return <main>Face attendance dashboard</main> }\n",
                encoding="utf-8",
            )

            task_plan = TaskPlan(
                implementation_steps=[
                    ImplementationTask(
                        filepath="src/App.jsx",
                        action="rewrite",
                        task_description="Build the face attendance app.",
                        dependencies=[],
                    )
                ]
            )
            state = {
                "original_user_prompt": "create a face detection attendance system",
                "plan": Plan(
                    name="Face Detection Attendance System",
                    description="A face detection attendance system for employee check-ins.",
                    techstack="React, Vite",
                    project_type="fullstack",
                    features=["Face Detection", "Attendance Tracking"],
                    files=[File(path="src/App.jsx", purpose="Main React component")],
                ),
                "task_plan": task_plan,
                "coder_state": CoderState(task_plan=task_plan, current_step_idx=1),
            }
            config = {"configurable": {"project_root": str(root)}}

            failure = graph._completion_guard_failure(state, config)

            self.assertIsNotNone(failure)
            self.assertEqual(failure.error_type, "MediaCaptureMissingError")

    def test_template_repair_for_face_attendance_adds_real_capture_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            src = root / "src"
            src.mkdir()

            (root / "package.json").write_text(
                json.dumps({"name": "react-vite", "type": "module"}),
                encoding="utf-8",
            )
            (root / "vite.config.js").write_text("export default {}\n", encoding="utf-8")
            (root / "index.html").write_text("<html><head></head><body><div id=\"root\"></div></body></html>\n", encoding="utf-8")
            (src / "main.jsx").write_text("import App from './App.jsx'\n", encoding="utf-8")
            (src / "App.jsx").write_text(
                "import viteLogo from '/vite.svg'\nconst text = 'Count is 0 and HMR is ready'\nexport default function App() { return <h1>Get started</h1> }\n",
                encoding="utf-8",
            )

            task_plan = TaskPlan(
                implementation_steps=[
                    ImplementationTask(
                        filepath="src/App.jsx",
                        action="rewrite",
                        task_description="Build the face attendance app.",
                        dependencies=[],
                    )
                ]
            )
            state = {
                "original_user_prompt": "create a face detection attendance system",
                "plan": Plan(
                    name="Face Detection Attendance System",
                    description="A face detection attendance system for employee check-ins.",
                    techstack="React, Vite",
                    project_type="fullstack",
                    features=["Face Detection", "Attendance Tracking"],
                    files=[File(path="src/App.jsx", purpose="Main React component")],
                ),
                "task_plan": task_plan,
                "coder_state": CoderState(task_plan=task_plan, current_step_idx=1),
            }
            config = {"configurable": {"project_root": str(root)}}

            failure = graph._completion_guard_failure(state, config)
            app_content = (src / "App.jsx").read_text(encoding="utf-8")

            self.assertIsNone(failure)
            self.assertIn("navigator.mediaDevices.getUserMedia", app_content)
            self.assertIn('type="file"', app_content)
            self.assertIn('accept="image/*"', app_content)


if __name__ == "__main__":
    unittest.main()
