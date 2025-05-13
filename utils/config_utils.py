import os
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Utility class for loading and accessing configuration settings.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigManager with the specified config file path.
        
        Args:
            config_path: Path to the config YAML file. If None, uses default path.
        """
        if config_path is None:
            # Get the project root directory
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(root_dir, "config", "config.yaml")
        
        # Load configuration
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the config YAML file.
            
        Returns:
            Dict containing the configuration.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: The configuration key (can be dot-separated for nested access)
            default: Default value to return if key doesn't exist
            
        Returns:
            The configuration value or default if not found
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using a sequence of keys.
        
        Args:
            *keys: Sequence of keys to navigate the nested structure
            default: Default value to return if path doesn't exist
            
        Returns:
            The configuration value or default if not found
        """
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value


# Global config manager instance
_config_manager = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get or create the global config manager instance.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager 