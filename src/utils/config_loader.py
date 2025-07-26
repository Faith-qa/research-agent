import yaml
from typing import List, Dict

def load_config() -> Dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
