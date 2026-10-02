"""Configuration loader and manager for Vantara Customer Intelligence Platform."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Load environment variables from .env if present
load_dotenv()


class ConfigManager:
    """Singleton-style manager to load and access project YAML configuration."""

    _instance: Optional["ConfigManager"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str] = None) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: Optional[str] = None) -> None:
        """Loads configuration from YAML file path."""
        path = config_path or os.getenv("CONFIG_PATH", "config/config.yaml")
        resolved_path = Path(path).resolve()

        if not resolved_path.exists():
            logger.warning(
                f"Config file not found at {resolved_path}. Using empty configuration."
            )
            self._config = {}
            return

        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration successfully from {resolved_path}")
        except Exception as e:
            logger.error(f"Failed to parse config YAML at {resolved_path}: {e}")
            raise

    @classmethod
    def get_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Returns the full configuration dictionary."""
        instance = cls(config_path)
        return instance._config

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value by key with optional default.

        Supports dot-notation keys, e.g. 'paths.raw_data_dir'.
        """
        instance = cls()
        keys = key.split(".")
        val: Any = instance._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Helper function to obtain config dictionary."""
    return ConfigManager.get_config(config_path)
