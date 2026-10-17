from pathlib import Path

import pytest

from acquisition import paths


@pytest.fixture(autouse=True)
def fresh_cache():
    paths.data_root.cache_clear()
    yield
    paths.data_root.cache_clear()


def test_env_var_is_honoured_as_is(tmp_path, monkeypatch):
    monkeypatch.setenv("CONFOCAL_MCP_DATA_DIR", str(tmp_path))
    assert paths.data_root() == tmp_path.resolve()
    assert paths.captures_dir() == tmp_path.resolve() / "data" / "captures"
    assert paths.logs_dir() == tmp_path.resolve() / "logs"


def test_unset_uses_a_writable_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("CONFOCAL_MCP_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert paths.data_root() == Path.cwd()


def test_unset_falls_back_when_cwd_is_not_writable(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CONFOCAL_MCP_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(paths, "_is_writable", lambda p: False)
    assert paths.data_root() == paths.FALLBACK_DATA_ROOT
    err = capsys.readouterr().err
    assert "not usable for runtime data" in err


def test_system_directory_is_rejected_even_if_writable(tmp_path, monkeypatch):
    monkeypatch.setenv("SystemRoot", str(tmp_path / "windows"))
    assert paths._is_system_dir(tmp_path / "windows")
    assert paths._is_system_dir(tmp_path / "windows" / "system32")
    assert not paths._is_system_dir(tmp_path / "other")
    monkeypatch.delenv("SystemRoot")
    assert not paths._is_system_dir(tmp_path)


def test_usable_data_root_rejects_system_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    assert paths._usable_data_root(tmp_path / "system32") == paths.FALLBACK_DATA_ROOT


def test_is_writable_probes_with_a_real_file(tmp_path):
    assert paths._is_writable(tmp_path / "new" / "dir")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert not paths._is_writable(blocker / "sub")
