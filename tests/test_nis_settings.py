# This repo's settings for the nis-elements-mcp package (acquisition/_nis_settings.py).
from acquisition import _nis_settings
from nis_elements_mcp import settings


def _unset(monkeypatch, *names):
    for name in names:
        monkeypatch.setenv(name, "")            # recorded first, so the undo removes what the call sets
        monkeypatch.delenv(name)


def test_logs_and_captures_stay_in_this_repos_data_folder(tmp_path, monkeypatch):
    _unset(monkeypatch, "NIS_LOG_DIR", "NIS_CAPTURE_DIR", "CONFOCAL_NIS_CAPTURE_DIR")
    monkeypatch.setattr(_nis_settings.paths, "logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(_nis_settings.paths, "data_root", lambda: tmp_path)
    _nis_settings.use_repo_folders()
    assert settings.log_dir() == tmp_path / "logs" / "nis"
    assert settings.capture_dir() == tmp_path / "data" / "NIS" / "captures"


def test_a_stop_engaged_in_the_old_file_is_moved_not_lost(tmp_path, monkeypatch):
    old, new = tmp_path / "old" / "ESTOP", tmp_path / "new" / "ESTOP"
    old.parent.mkdir()
    old.write_text('{"reason": "pressed before the switch"}')
    monkeypatch.setenv("NIS_ESTOP_FILE", str(new))
    _nis_settings.carry_over_old_estop(old)
    assert not old.exists() and "pressed before the switch" in new.read_text()


def test_an_engaged_new_file_is_kept_and_the_old_one_dropped(tmp_path, monkeypatch):
    old, new = tmp_path / "ESTOP.old", tmp_path / "ESTOP"
    old.write_text("old")
    new.write_text("new")
    monkeypatch.setenv("NIS_ESTOP_FILE", str(new))
    _nis_settings.carry_over_old_estop(old)
    assert not old.exists() and new.read_text() == "new"


def test_nothing_happens_without_an_old_file(tmp_path, monkeypatch):
    monkeypatch.setenv("NIS_ESTOP_FILE", str(tmp_path / "ESTOP"))
    _nis_settings.carry_over_old_estop(tmp_path / "missing")
    assert not (tmp_path / "ESTOP").exists()


def test_old_switch_names_still_work(monkeypatch):
    _unset(monkeypatch, "NIS_NO_ESTOP_PANEL", "NIS_NO_MONITOR", "NIS_BRIDGE_HOST", "NIS_BRIDGE_PORT")
    monkeypatch.setenv("CONFOCAL_NO_ESTOP_PANEL", "1")
    monkeypatch.setenv("CONFOCAL_NO_NIS_MONITOR", "1")
    monkeypatch.setenv("CONFOCAL_NIS_BRIDGE_URL", "http://127.0.0.1:9001")
    _nis_settings.use_old_names()
    assert settings.estop_panel_disabled() and settings.monitor_disabled()
    assert settings.bridge_url() == "http://127.0.0.1:9001"
