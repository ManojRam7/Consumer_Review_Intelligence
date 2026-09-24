from review_intel.report import END, START, update_readme


def test_readme_block_is_replaced(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{START}\nold\n{END}\n\nFooter\n")
    assert update_readme("new table", readme)
    text = readme.read_text()
    assert "new table" in text and "old" not in text and text.endswith("Footer\n")


def test_missing_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n")
    assert not update_readme("x", readme)
