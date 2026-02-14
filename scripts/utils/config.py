"""
Configuration Manager
Loads and manages YAML configuration files
"""
import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional

class Config:
    """
    Configuration loader and manager
    
    Usage:
        config = Config()
        batch_size = config.get('data.batch_size')
        ear_threshold = config['detection.ear_threshold']
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration
        
        Args:
            config_path: Path to main config file
        """
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        
        # Load main config
        self.config = self._load_yaml(self.config_path)
        
        # Load additional configs
        self.model_config = self._load_yaml(self.config_dir / "model_config.yaml")
        self.paths = self._load_yaml(self.config_dir / "paths.yaml")
    
    def _load_yaml(self, path: Path) -> Dict:
        """Load YAML file"""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def get(self, key: str, default: Any = None, config_type: str = "main") -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Key in dot notation (e.g., 'data.batch_size')
            default: Default value if key not found
            config_type: Which config to use ('main', 'model', 'paths')
        
        Returns:
            Configuration value
        
        Example:
            >>> config.get('data.batch_size')
            32
            >>> config.get('detection.ear_threshold')
            0.25
        """
        # Select config
        if config_type == "model":
            cfg = self.model_config
        elif config_type == "paths":
            cfg = self.paths
        else:
            cfg = self.config
        
        # Navigate nested keys
        keys = key.split('.')
        value = cfg
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key not found: {key}")
        return value
    
    def get_path(self, key: str) -> Path:
        """
        Get path from paths config
        
        Args:
            key: Path key in dot notation
        
        Returns:
            Path object
        
        Example:
            >>> config.get_path('models.saved')
            Path('models/saved')
        """
        path_str = self.get(key, config_type="paths")
        if path_str is None:
            raise KeyError(f"Path not found: {key}")
        return Path(path_str)
    
    def get_model_config(self, model_name: str) -> Dict:
        """
        Get model-specific configuration
        
        Args:
            model_name: Name of model ('eye_classifier' or 'yawn_detector')
        
        Returns:
            Model configuration dict
        """
        return self.model_config.get(model_name, {})
    
    def create_directories(self):
        """Create all necessary directories from paths config"""
        paths_to_create = [
            'data.raw.eyes.train',
            'data.raw.eyes.val',
            'data.raw.eyes.test',
            'data.raw.yawns.train',
            'data.raw.yawns.val',
            'data.processed.root',
            'data.augmented.root',
            'models.saved',
            'models.checkpoints',
            'models.tflite',
            'outputs.logs.root',
            'outputs.plots.root',
            'outputs.reports.root',
        ]
        
        for path_key in paths_to_create:
            try:
                path = self.get_path(path_key)
                path.mkdir(parents=True, exist_ok=True)
            except KeyError:
                continue
        
        print("✓ All directories created successfully")
    
    def validate(self) -> bool:
        """
        Validate configuration values
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check data splits sum to 1
        train = self.get('data.train_split', 0)
        val = self.get('data.val_split', 0)
        test = self.get('data.test_split', 0)
        
        if abs(train + val + test - 1.0) > 0.01:
            raise ValueError(
                f"Data splits must sum to 1.0, got {train + val + test}"
            )
        
        # Check positive values
        if self.get('data.batch_size', 0) <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.get('detection.ear_threshold', 0) <= 0:
            raise ValueError("EAR threshold must be positive")
        
        return True
    
    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"


# Global config instance
config = Config()


if __name__ == "__main__":
    # Test configuration loading
    print("="*70)
    print("CONFIGURATION TEST")
    print("="*70)
    print()
    
    # Test basic access
    print("Project name:", config.get('project.name'))
    print("Batch size:", config.get('data.batch_size'))
    print("EAR threshold:", config.get('detection.ear_threshold'))
    print()
    
    # Test dict-like access
    print("Learning rate:", config['training.eye_classifier.learning_rate'])
    print()
    
    # Test paths
    print("Models path:", config.get_path('models.saved'))
    print("Logs path:", config.get_path('outputs.logs.root'))
    print()
    
    # Test model config
    eye_config = config.get_model_config('eye_classifier')
    print(f"Eye classifier layers: {len(eye_config.get('layers', []))}")
    print()
    
    # Validate
    try:
        config.validate()
        print("✓ Configuration valid")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
    
    print()
    print("="*70)