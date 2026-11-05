import json
import pathlib
import tempfile
import unittest

import graph
from schemas import File, Plan


def tailwind_plan() -> Plan:
    return Plan(
        name="StudyFlow",
        description="A study dashboard using Tailwind CSS for layout.",
        techstack="React, Vite, Tailwind CSS",
        project_type="react-vite",
        features=["Task tracking", "Tailwind styling"],
        files=[
            File(path="src/App.jsx", purpose="Main React app"),
            File(path="src/index.css", purpose="Tailwind CSS entry file"),
        ],
    )


class TailwindHardeningTests(unittest.TestCase):
    def test_fallback_requirements_add_tailwind_toolchain(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec = graph._fallback_requirements_spec(tailwind_plan(), pathlib.Path(tmp_dir), "react-vite")

        self.assertEqual(spec.package_manager, "npm")
        self.assertIn("tailwindcss@^3.4.19", spec.packages)
        self.assertIn("postcss@^8.5.15", spec.packages)
        self.assertIn("autoprefixer@^10.5.0", spec.packages)

    def test_ensure_tailwind_setup_creates_configs_and_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            (root / "src").mkdir()
            (root / "package.json").write_text(
                json.dumps(
                    {
                        "name": "studyflow",
                        "type": "module",
                        "dependencies": {"react": "^18.3.1"},
                        "devDependencies": {"vite": "^5.4.0"},
                    }
                ),
                encoding="utf-8",
            )

            graph._ensure_tailwind_setup(root, tailwind_plan())

            package_data = json.loads((root / "package.json").read_text(encoding="utf-8"))
            dev_dependencies = package_data["devDependencies"]
            self.assertEqual(dev_dependencies["tailwindcss"], "^3.4.19")
            self.assertEqual(dev_dependencies["postcss"], "^8.5.15")
            self.assertEqual(dev_dependencies["autoprefixer"], "^10.5.0")
            self.assertIn("src/**/*.{js,jsx,ts,tsx}", (root / "tailwind.config.js").read_text(encoding="utf-8"))
            self.assertIn("tailwindcss", (root / "postcss.config.js").read_text(encoding="utf-8"))

    def test_missing_tailwind_configs_fail_verification(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            (root / "src").mkdir()
            (root / "package.json").write_text(
                json.dumps(
                    {
                        "name": "studyflow",
                        "type": "module",
                        "dependencies": {"react": "^18.3.1"},
                        "devDependencies": {"tailwindcss": "^3.4.19"},
                    }
                ),
                encoding="utf-8",
            )
            (root / "PRD.md").write_text("Use Tailwind CSS for the dashboard.\n", encoding="utf-8")
            (root / "src" / "index.css").write_text("@tailwind base;\n", encoding="utf-8")

            failure = graph._tailwind_setup_failure(root)

            self.assertIsNotNone(failure)
            self.assertEqual(failure.error_type, "TailwindSetupError")
            self.assertIn("postcss", failure.stderr)
            self.assertIn("autoprefixer", failure.stderr)
            self.assertIn("tailwind.config.js", failure.stderr)
            self.assertIn("postcss.config.js", failure.stderr)

    def test_raw_tailwind_directives_in_built_css_fail_verification(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            assets = root / "dist" / "assets"
            assets.mkdir(parents=True)
            (root / "package.json").write_text(
                json.dumps(
                    {
                        "name": "studyflow",
                        "type": "module",
                        "dependencies": {"react": "^18.3.1"},
                        "devDependencies": {
                            "tailwindcss": "^3.4.19",
                            "postcss": "^8.5.15",
                            "autoprefixer": "^10.5.0",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "tailwind.config.js").write_text(
                "export default { content: ['./index.html', './src/**/*.{js,jsx}'] }\n",
                encoding="utf-8",
            )
            (root / "postcss.config.js").write_text(
                "export default { plugins: { tailwindcss: {}, autoprefixer: {} } }\n",
                encoding="utf-8",
            )
            (assets / "index.css").write_text("@tailwind base;\n.button{@apply font-bold;}\n", encoding="utf-8")

            failure = graph._tailwind_setup_failure(root)

            self.assertIsNotNone(failure)
            self.assertEqual(failure.error_type, "TailwindBuildError")
            self.assertIn("Raw @tailwind/@apply directives", failure.stderr)
            self.assertIn("dist/assets/index.css", failure.stderr)


if __name__ == "__main__":
    unittest.main()
