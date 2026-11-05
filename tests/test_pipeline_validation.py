import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from pipeline_validation import (
    build_generated_file_manifest,
    placeholder_failure_result,
    scan_placeholders,
    run_runtime_health_check,
    validate_project_imports,
)
from prompt import PLACEHOLDER_PATTERNS


class PipelineValidationTests(unittest.TestCase):
    def test_manifest_extracts_js_and_python_facts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "Button.jsx").write_text(
                "import helper from './helper';\n"
                "export default function Button() { return null }\n"
                "export const buttonSize = 'sm';\n",
                encoding="utf-8",
            )
            (root / "src" / "calc.py").write_text(
                "import os\n"
                "from .helpers import add\n\n"
                "class Calculator: pass\n"
                "def total(): return 1\n",
                encoding="utf-8",
            )

            manifest = build_generated_file_manifest(root)

            self.assertIn("src/Button.jsx", manifest)
            self.assertIn("./helper", manifest["src/Button.jsx"].imports)
            self.assertIn("default", manifest["src/Button.jsx"].exports)
            self.assertIn("buttonSize", manifest["src/Button.jsx"].exports)
            self.assertIn("Calculator", manifest["src/calc.py"].exports)
            self.assertIn("total", manifest["src/calc.py"].exports)

    def test_placeholder_scan_reports_offending_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text("def run():\n    raise NotImplementedError\n", encoding="utf-8")

            hits = scan_placeholders(root, PLACEHOLDER_PATTERNS)
            result = placeholder_failure_result(hits)

            self.assertIsNotNone(result)
            self.assertIn("app.py:2", result.stderr)
            self.assertEqual(result.error_type, "PlaceholderError")

    def test_validate_project_imports_fails_on_missing_relative_import(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "App.jsx").write_text(
                "import Missing from './Missing';\nexport default function App() { return <Missing /> }\n",
                encoding="utf-8",
            )

            result = validate_project_imports(root)

            self.assertIsNotNone(result)
            self.assertEqual(result.error_type, "ImportValidationError")
            self.assertIn("./Missing", result.stderr)

    def test_validate_project_imports_allows_existing_asset_import(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "style.css").write_text("body {}\n", encoding="utf-8")
            (root / "src" / "App.jsx").write_text("import './style.css';\n", encoding="utf-8")

            self.assertIsNone(validate_project_imports(root))

    def test_runtime_health_check_reports_early_process_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_plan = {
                "commands": [{"name": "Frontend", "cmd": "python -c \"import sys; sys.exit(3)\""}],
                "serve_port": 3999,
            }
            with patch("pipeline_validation.build_run_plan", return_value=fake_plan):
                result = run_runtime_health_check(root, timeout_seconds=2)

            self.assertIsNotNone(result)
            self.assertIn(result.error_type, {"RuntimeStartError", "RuntimeHealthCheckError"})


if __name__ == "__main__":
    unittest.main()
