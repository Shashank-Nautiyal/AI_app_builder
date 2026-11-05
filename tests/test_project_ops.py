import pathlib
import tempfile
import unittest
import zipfile

import project_ops
import project_runner


class ProjectOpsTests(unittest.TestCase):
    def test_extract_project_zip_strips_single_root_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = pathlib.Path(tmp_dir)
            archive = base / "project.zip"
            target = base / "out"

            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("sample-app/src/main.py", "print('hello')\n")
                zf.writestr("sample-app/README.md", "# demo\n")

            extracted = project_ops.extract_project_zip(archive, target)

            self.assertIn("src/main.py", extracted)
            self.assertTrue((target / "src" / "main.py").exists())
            self.assertFalse((target / "sample-app").exists())

    def test_import_project_files_strips_common_root_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = pathlib.Path(tmp_dir)
            target = base / "out"

            imported = project_ops.import_project_files(
                [
                    ("demo-app/src/main.py", b"print('hello')\n"),
                    ("demo-app/README.md", b"# demo\n"),
                ],
                target,
            )

            self.assertIn("src/main.py", imported)
            self.assertTrue((target / "src" / "main.py").exists())
            self.assertFalse((target / "demo-app").exists())

    def test_import_project_files_preserves_direct_file_uploads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = pathlib.Path(tmp_dir)
            target = base / "out"

            imported = project_ops.import_project_files(
                [
                    ("app.py", b"print('hello')\n"),
                    ("requirements.txt", b"fastapi\n"),
                ],
                target,
            )

            self.assertEqual(sorted(imported), ["app.py", "requirements.txt"])
            self.assertTrue((target / "app.py").exists())
            self.assertTrue((target / "requirements.txt").exists())

    def test_create_and_restore_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = pathlib.Path(tmp_dir)
            project = base / "project"
            project.mkdir()
            source = project / "app.py"
            source.write_text("print('v1')\n", encoding="utf-8")

            snapshot = project_ops.create_project_snapshot(project)
            source.write_text("print('v2')\n", encoding="utf-8")

            project_ops.restore_project_snapshot(project, snapshot)

            self.assertEqual(source.read_text(encoding="utf-8"), "print('v1')\n")

    def test_dependency_name_handles_scoped_npm_packages(self):
        self.assertEqual(
            project_runner._dependency_name("@supabase/supabase-js@^2", "npm"),
            "@supabase/supabase-js",
        )
        self.assertEqual(
            project_runner._dependency_name("@clerk/clerk-react", "npm"),
            "@clerk/clerk-react",
        )

    def test_project_runner_detects_new_templates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = pathlib.Path(tmp_dir)

            streamlit_dir = base / "streamlit"
            streamlit_dir.mkdir()
            (streamlit_dir / "app.py").write_text("import streamlit as st\n", encoding="utf-8")
            self.assertEqual(project_runner.ProjectRunner(streamlit_dir).detect_type(), "streamlit-app")

            tkinter_dir = base / "tkinter"
            tkinter_dir.mkdir()
            (tkinter_dir / "main.py").write_text("import tkinter as tk\n", encoding="utf-8")
            (tkinter_dir / "database.py").write_text("import sqlite3\n", encoding="utf-8")
            self.assertEqual(project_runner.ProjectRunner(tkinter_dir).detect_type(), "python-tkinter-sqlite")

            electron_dir = base / "electron"
            (electron_dir / "electron").mkdir(parents=True)
            (electron_dir / "package.json").write_text(
                '{"devDependencies": {"electron": "^33.0.0"}}',
                encoding="utf-8",
            )
            self.assertEqual(project_runner.ProjectRunner(electron_dir).detect_type(), "electron-react-app")
