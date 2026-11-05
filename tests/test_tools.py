import pathlib
import unittest

import Tools


class ToolsSafetyTests(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path.cwd() / "tmp_test_root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.root.exists():
            for child in sorted(self.root.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    child.rmdir()
            self.root.rmdir()

    def test_safe_rejects_path_escape(self):
        with self.assertRaises(ValueError):
            Tools._safe("../outside.txt", self.root)

    def test_run_cmd_rejects_empty_command(self):
        result = Tools.run_cmd.invoke(
            {"cmd": "   ", "cwd": ".", "timeout": 5, "config": {"configurable": {"thread_id": "test"}}}
        )
        self.assertIn("ERROR: command is empty", result)

    def test_dependency_installs_use_host_subprocess_path(self):
        self.assertTrue(Tools._should_use_host_subprocess("npm install"))
        self.assertTrue(Tools._should_use_host_subprocess("pip install -r requirements.txt"))

    def test_apply_patch_block_handles_whitespace_differences(self):
        target = self.root / "app.py"
        target.write_text("def demo():\n    print('hello')\n", encoding="utf-8")

        patch = (
            "<<<<<<< SEARCH\n"
            "  print('hello')\n"
            "=======\n"
            "    print('updated')\n"
            ">>>>>>> REPLACE"
        )
        result = Tools.apply_patch_block.invoke(
            {"path": "app.py", "patch": patch},
            config={"configurable": {"project_root": str(self.root)}},
        )

        self.assertIn("PATCHED", result)
        self.assertIn("updated", target.read_text(encoding="utf-8"))

    def test_list_files_returns_real_newlines(self):
        (self.root / "a.txt").write_text("a", encoding="utf-8")
        (self.root / "nested").mkdir()
        (self.root / "nested" / "b.txt").write_text("b", encoding="utf-8")

        result = Tools.list_files.invoke(
            {"directory": "."},
            config={"configurable": {"project_root": str(self.root)}},
        )

        self.assertIn("a.txt\n", result)
        self.assertNotIn("\\n", result)

    def test_replace_file_content_uses_search_replace_engine(self):
        target = self.root / "service.py"
        target.write_text("if True:\n    return 'old'\n", encoding="utf-8")

        result = Tools.replace_file_content.invoke(
            {
                "path": "service.py",
                "target_content": " return 'old'",
                "replacement_content": "    return 'new'",
            },
            config={"configurable": {"project_root": str(self.root)}},
        )

        self.assertIn("PATCHED", result)
        self.assertIn("'new'", target.read_text(encoding="utf-8"))

    def test_delete_file_requires_reason_and_refuses_bulk_deletes(self):
        target = self.root / "stale.txt"
        directory = self.root / "nested"
        target.write_text("obsolete", encoding="utf-8")
        directory.mkdir()
        config = {"configurable": {"project_root": str(self.root)}}

        root_result = Tools.delete_file.invoke({"path": "."}, config=config)
        directory_result = Tools.delete_file.invoke(
            {"path": "nested", "reason": "testing directory delete protection"},
            config=config,
        )
        missing_reason_result = Tools.delete_file.invoke({"path": "stale.txt"}, config=config)
        delete_result = Tools.delete_file.invoke(
            {"path": "stale.txt", "reason": "removing explicitly named obsolete file"},
            config=config,
        )

        self.assertIn("Refusing to delete the project root", root_result)
        self.assertIn("Refusing to delete directory", directory_result)
        self.assertIn("requires a specific reason", missing_reason_result)
        self.assertIn("DELETED: stale.txt", delete_result)
        self.assertTrue(directory.exists())
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
