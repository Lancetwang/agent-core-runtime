import unittest
from tempfile import TemporaryDirectory

from agent_core.tools import build_edit_file_tool, build_read_file_tool, build_write_file_tool


class FileToolTests(unittest.TestCase):
    def test_write_read_and_edit_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            write_file = build_write_file_tool(temp_dir)
            read_file = build_read_file_tool(temp_dir)
            edit_file = build_edit_file_tool(temp_dir)

            write_result = write_file.execute(path="reports/a.md", content="one\ntwo\nthree\n")
            read_result = read_file.execute(path="reports/a.md", start_line=2, max_chars=100)
            edit_result = edit_file.execute(
                path="reports/a.md",
                start_line=2,
                end_line=2,
                replacement="TWO",
            )
            edited = read_file.execute(path="reports/a.md", start_line=1, max_chars=100)

        self.assertEqual(write_result["line_count"], 3)
        self.assertIn("2: two", read_result["content"])
        self.assertEqual(edit_result["replacement_line_count"], 1)
        self.assertIn("2: TWO", edited["content"])

    def test_file_tools_reject_path_escape(self) -> None:
        with TemporaryDirectory() as temp_dir:
            write_file = build_write_file_tool(temp_dir)

            with self.assertRaises(ValueError):
                write_file.execute(path="../outside.md", content="bad")


if __name__ == "__main__":
    unittest.main()
