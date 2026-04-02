"""
nova.fmt — Format serialization constellation
Handles JSON, YAML, TOML serialization and deserialization.
"""

import json
from typing import Any, Dict

__all__ = [
    'json_dump', 'json_load',
    'yaml_dump', 'yaml_load',
    'toml_dump', 'toml_load'
]


def json_dump(obj: Any) -> str:
    """
    Serialize object to JSON string.
    
    Args:
        obj: Object to serialize (dict, list, etc.)
    
    Returns:
        JSON string (nova: String)
    """
    return json.dumps(obj, indent=2)


def json_load(data: str) -> Dict[str, Any]:
    """
    Deserialize JSON string to object.
    
    Args:
        data: JSON string
    
    Returns:
        Deserialized object (nova: Struct or Array)
    """
    return json.loads(data)


def yaml_dump(obj: Any) -> str:
    """
    Serialize object to YAML string.
    Requires PyYAML library.
    
    Args:
        obj: Object to serialize
    
    Returns:
        YAML string (nova: String)
    """
    try:
        import yaml
        return yaml.dump(obj, default_flow_style=False)
    except ImportError:
        raise ImportError("PyYAML required for YAML support: pip install pyyaml")


def yaml_load(data: str) -> Any:
    """
    Deserialize YAML string to object.
    Requires PyYAML library.
    
    Args:
        data: YAML string
    
    Returns:
        Deserialized object
    """
    try:
        import yaml
        return yaml.safe_load(data)
    except ImportError:
        raise ImportError("PyYAML required for YAML support: pip install pyyaml")


def toml_dump(obj: Any) -> str:
    """
    Serialize object to TOML string.
    Requires tomli_w library.
    
    Args:
        obj: Object to serialize (must be dict)
    
    Returns:
        TOML string (nova: String)
    """
    try:
        import tomli_w
        return tomli_w.dumps(obj)
    except ImportError:
        raise ImportError("tomli_w required for TOML support: pip install tomli_w")


def toml_load(data: str) -> Dict[str, Any]:
    """
    Deserialize TOML string to object.
    Requires tomli library.
    
    Args:
        data: TOML string
    
    Returns:
        Deserialized object (dict)
    """
    try:
        import tomli
        return tomli.loads(data)
    except ImportError:
        raise ImportError("tomli required for TOML support: pip install tomli")
