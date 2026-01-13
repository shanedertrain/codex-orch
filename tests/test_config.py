from pathlib import Path

from codex_orch.config import default_config, load_config


def test_loads_default_config(tmp_path: Path) -> None:
    repo = tmp_path
    config_path = repo / ".orchestrator" / "orchestrator.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("")

    # Write default config then reload through loader
    config_path.write_text("")
    config_data = default_config()
    import yaml

    config_path.write_text(yaml.safe_dump(config_data))
    config, paths = load_config(config_path, repo)

    assert paths.decisions == repo / "docs/ai/decisions.md"
    assert config.role_for("navigator") is not None
    assert (
        config.role_for("implementer").output_schema.name == "task_result.schema.json"
    )
    assert config.use_single_workspace is True
