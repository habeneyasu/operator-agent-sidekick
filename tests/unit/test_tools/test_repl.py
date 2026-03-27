from src.tools.repl import run_python_code, repl_tool


def test_run_python_prints_stdout():
    out = run_python_code("print(2 + 2)")
    assert "4" in out


def test_run_python_timeout():
    out = run_python_code("while True: pass", timeout_seconds=1)
    assert "timeout" in out.lower()


def test_repl_tool_invoke():
    tool = repl_tool()
    assert tool.name == "run_python"
    assert "2" in tool.run("print(1+1)")
