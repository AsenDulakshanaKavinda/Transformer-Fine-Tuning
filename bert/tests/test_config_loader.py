import os
import tempfile
import yaml
import pytest                
from pathlib import Path

from src.utils.config_loader import load_config, _project_root


def test_load_config_with_explicit_path(tmp_path):
    # Create a temporary YAML file
    config_data = {"model": {"name": "bert-base-uncased"}}
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    # Call load_config with explicit path
    result = load_config(str(config_file))

    # Assert
    assert result == config_data


def test_load_config_with_env_var(tmp_path, monkeypatch):
    # Create temporary YAML file
    config_data = {"train": {"batch_size": 32}}
    config_file = tmp_path / "env_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    # Set CONFIG_PATH environment variable
    monkeypatch.setenv("CONFIG_PATH", str(config_file))

    result = load_config()

    assert result == config_data


def test_load_config_file_not_found(tmp_path):
    # Non-existent file path
    fake_path = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(str(fake_path))


def test_load_config_returns_empty_if_file_empty(tmp_path):
    # Empty YAML file
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")

    result = load_config(str(empty_file))
    assert result == {}  # safe_load returns None, we handle as {}


def test_project_root_is_correct():
    # Check if _project_root points to expected location (up 3 directories)
    root = _project_root()
    assert root.exists()
    assert isinstance(root, Path)
