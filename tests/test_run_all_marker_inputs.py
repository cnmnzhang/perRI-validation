"""The all runner accepts complete, site-provided marker files."""

import run_all


def test_all_uses_supplied_marker_files_without_master(tmp_path, monkeypatch, capsys):
    marker_dir = tmp_path / "data" / "cache" / "splits_by_marker"
    marker_dir.mkdir(parents=True)
    (marker_dir / "HB.csv").write_text("anon_id,ts,test_code,result_value,sex\n")
    called = []
    monkeypatch.setattr(run_all, "ANALYSES", {
        "splits_by_marker": lambda **kwargs: called.append("split"),
        "setpoints_by_marker": lambda **kwargs: called.append("fit"),
    })

    run_all.main(["--analysis", "all", "--input-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path / "out")])

    assert called == ["fit"]
    assert "using site-provided marker files" in capsys.readouterr().out


def test_all_builds_split_when_master_is_present(tmp_path, monkeypatch):
    input_dir = tmp_path / "data"
    marker_dir = input_dir / "cache" / "splits_by_marker"
    marker_dir.mkdir(parents=True)
    (marker_dir / "HB.csv").write_text("anon_id,ts,test_code,result_value,sex\n")
    (input_dir / "tests.csv").write_text("anon_id,ts,test_code,result_value,sex\n")
    called = []
    monkeypatch.setattr(run_all, "ANALYSES", {
        "splits_by_marker": lambda **kwargs: called.append("split"),
        "setpoints_by_marker": lambda **kwargs: called.append("fit"),
    })

    run_all.main(["--analysis", "all", "--input-dir", str(input_dir), "--output-dir", str(tmp_path / "out")])

    assert called == ["split", "fit"]
