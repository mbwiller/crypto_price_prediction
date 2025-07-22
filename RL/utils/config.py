import yaml
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
