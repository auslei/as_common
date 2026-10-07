import logging
import sys
import yaml
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

CONFIG_PATH = Path("config.yaml")

_config_logger = logging.getLogger(__name__)

class LLMConfig(BaseModel):
    provider: str = Field(default="ollama")
    model: str = Field(default="qwen3:14b")
    temperature: float = Field(default=0.0)
    context_window: Optional[int] = Field(default=None)
    concurrency: int = Field(default=2)

class AppConfig(BaseModel):
    api_key: str = Field(default="dev-secret-key")
    llm: LLMConfig = Field(default_factory=LLMConfig)

def load_config_from_path(config_path: Path) -> AppConfig:
    """Loads application configuration from a specific YAML file."""
    logging.info(f"Loading config from: {config_path}")
    if not config_path.exists():
        print(f"[config] WARNING: config file not found: {config_path}", file=sys.stderr)
        return AppConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = AppConfig(**(data or {}))
        print(f"[config] Loaded: {config_path}", file=sys.stderr)
        return cfg
    except Exception as e:
        print(f"[config] FATAL: failed to parse {config_path}: {e}", file=sys.stderr)
        raise

def find_config() -> Path:
    """
    Determines which config file to use based on environment.
    Priority:
    1. PACKAGE_CONFIG_PATH env var
    2. packages/{PACKAGE_NAME}/config.yaml
    3. src/{PACKAGE_NAME}/config.yaml
    4. Root config.yaml
    """
    env_path = os.getenv("PACKAGE_CONFIG_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        _config_logger.warning("PACKAGE_CONFIG_PATH=%s does not exist — falling back", env_path)

    pkg_name = os.getenv("PACKAGE_NAME") or os.getenv("INVESTOR_IQ_PACKAGE")
    if pkg_name:
        pkg_cfg = Path("packages") / pkg_name / "config.yaml"
        if pkg_cfg.exists():
            return pkg_cfg

        src_cfg = Path("src") / pkg_name / "config.yaml"
        if src_cfg.exists():
            return src_cfg

        _config_logger.warning("PACKAGE_NAME=%s set but no config.yaml found — falling back to root", pkg_name)

    return CONFIG_PATH

def load_config() -> AppConfig:
    """Loads the appropriate configuration for the current context."""
    return load_config_from_path(find_config())

def reload_config():
    """Reloads the global configuration instance."""
    global config
    config = load_config()

# Global configuration instance
config = load_config()
