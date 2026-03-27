from pathlib import Path

from src.tools.file_manager import (
    list_sandbox_dir,
    read_sandbox_file,
    write_sandbox_file,
)
from src.tools.file_manager import file_manager_tools
from src.tools.spec import SidekickTool


def test_resolve_rejects_escape(tmp_path: Path):
    root = tmp_path / "s"
    root.mkdir()
    try:
        read_sandbox_file(root, "../outside")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "sandbox" in str(exc).lower() or "escape" in str(exc).lower()


def test_read_write_list_roundtrip(tmp_path: Path):
    root = tmp_path / "sb"
    root.mkdir()
    assert "Wrote" in write_sandbox_file(root, "a/b.txt", "hello")
    assert read_sandbox_file(root, "a/b.txt") == "hello"
    listing = list_sandbox_dir(root, ".")
    assert "a/" in listing


def test_file_manager_tools_invoke(tmp_path: Path):
    tools = file_manager_tools(tmp_path / "sb")
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {
        "read_sandbox_file",
        "write_sandbox_file",
        "list_sandbox_dir",
    }
    assert "Wrote" in by_name["write_sandbox_file"].run(
        '{"path": "x.txt", "content": "hi"}'
    )
    assert by_name["read_sandbox_file"].run('{"path": "x.txt"}') == "hi"
    assert "x.txt" in by_name["list_sandbox_dir"].run("")


def test_sidekick_tool_is_frozen_dataclass():
    t = SidekickTool(name="n", description="d", invoke=lambda s: s)
    assert t.run("x") == "x"
