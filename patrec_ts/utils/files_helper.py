from abc import ABC, abstractmethod

from typing import Dict, Any, Literal, Optional, Union, List, Self
from pathlib import Path
import json
import ast
from datetime import datetime
from collections.abc import Mapping

import numpy as np

__all__ =  [
    # loaders
    'TS_baseloader',
    'Format1_loader',
    # data upload functions
    'JSON_Handler',
    'YAML_Config',
    'load_yaml_file',
    'load_config_file',
    'save_config_file',
    ]


class TS_baseloader(ABC):
    """
    Base class for time series data loading
    """
    def __init__(self, filepath: str = None, data: dict = None):
        """
        Initialize the loader either from a file or from a data dictionary
        
        Args:
            filepath: Path to the data file
            data: Dictionary/ Arrays containing time series data
        """
        if filepath is not None:
            self._load_data(filepath)
        elif data is not None:
            self._update_data(data)
        else:
            raise ValueError("Either filepath or data must be provided")

    @abstractmethod
    def _load_data(self, filepath: Union[str, Path]) -> None:
        """Load data from file"""
        pass
    
    @abstractmethod
    def _update_data(self, data: dict):
        """Update data from dictionary"""
        pass


class Format1_loader(TS_baseloader):
    """
    Loader for Format1 time series data structure:
    {
        'data': np.ndarray of time series (will be converted from list if needed),
        'labels': np.ndarray or list of class labels,
        'meta': list[dict] of metadata dictionaries
    }
    """
    def __init__(self, filepath: str = None, data: dict = None, 
                 npy_files: dict = None):
        """
        Initialize the loader from:
        - JSON file (filepath)
        - Data dictionary (data)
        - Three NPY files (npy_files)
        
        Args:
            filepath: Path to JSON file containing all data
            data: Dictionary containing 'data', 'labels', and 'meta'
            npy_files: Dictionary with paths to:
                - 'data': NPY file with time series data
                - 'labels': NPY file with labels
                - 'meta': NPY file with metadata (will be converted to list[dict])
        """
        if npy_files is not None:
            self._load_npy_files(npy_files)
        else:
            super().__init__(filepath, data)
        self._validate_structure()

    def _load_data(self, filepath: str):
        """
        Load data from JSON file and ensure proper types:
        - Converts data to np.ndarray
        - Keeps meta as list[dict]
        - Preserves labels type (list or np.ndarray)
        """
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        self._update_data(loaded_data)

    def _load_npy_files(self, npy_files: dict):
        """
        Load data from three separate NPY files:
        - data: NPY file with time series data
        - labels: NPY file with labels
        - meta: NPY file with metadata (will be converted to list[dict])
        
        Args:
            npy_files: Dictionary containing paths to:
                'data': path to data NPY file
                'labels': path to labels NPY file
                'meta': path to metadata NPY file
        """
        if not all(key in npy_files for key in ['data', 'labels', 'meta']):
            raise ValueError("npy_files must contain 'data', 'labels', and 'meta' keys")
            
        # Load numpy arrays from files
        try:
            data = np.load(npy_files['data'])
            labels = np.load(npy_files['labels'])
            meta = np.load(npy_files['meta'], allow_pickle=True) if npy_files['meta'] is not None else [{} for _ in range(data.shape[0])] 
        except Exception as e:
            raise ValueError(f"Error loading NPY files: {str(e)}")
            
        # Convert metadata to list of dictionaries if needed
        if isinstance(meta, np.ndarray):
            meta = meta.tolist()
            
        if not isinstance(meta, list) or not all(isinstance(item, dict) for item in meta):
            raise TypeError("Loaded metadata must be convertible to list[dict]")
            
        self._update_data({
            'data': data,
            'labels': labels,
            'meta': meta
        })

    def _update_data(self, data: dict):
        """
        Update data from dictionary with strict type conversion:
        - data always becomes np.ndarray
        - meta must be list[dict]
        """
        if not all(key in data for key in ['data', 'labels', 'meta']):
            raise ValueError("Data must contain 'data', 'labels', and 'meta' keys")
            
        # Force data to be numpy array
        self.data = np.array(data['data'])
        
        # Preserve original labels type (list or array)
        self.labels = data['labels']  
         
        # Validate and store meta
        if not isinstance(data['meta'], list):
            raise TypeError("'meta' must be a list")
        self.meta = data['meta']
        
        self._validate_structure()

    def _validate_structure(self):
        """
        Validate the complete data structure:
        - data must be np.ndarray
        - labels must be list or np.ndarray
        - meta must be list[dict]
        - lengths must match
        """
        if not isinstance(self.data, np.ndarray):
            raise TypeError("'data' must be numpy.ndarray")
            
        if not isinstance(self.labels, (np.ndarray, list)):
            raise TypeError("'labels' must be numpy array or list")
            
        if not (isinstance(self.meta, list) and all(isinstance(item, dict) for item in self.meta)):
            raise TypeError("'meta' must be a list of dictionaries")
            
        if len(self.data) != len(self.labels) or len(self.data) != len(self.meta):
            raise ValueError("Length of 'data', 'labels' and 'meta' must match")

    def to_json(self, filepath: str):
        """
        Save data to JSON with proper type conversion:
        - data converted to list
        - labels converted to list if needed
        - meta remains as list[dict]
        """
        save_data = {
            'data': self.data.tolist(),
            'labels': self.labels.tolist() if isinstance(self.labels, np.ndarray) else self.labels,
            'meta': self.meta 
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)

    @property
    def shape(self) -> tuple:
        """Return shape of the time series data"""
        return self.data.shape


class JSON_Handler:
    def __init__(
        self,
        filepath: str | Path | None = None,
        data: Optional[Union[Dict, List]] = None,
        mode: Literal['write', 'append', 'update'] = 'write',
        **kwargs
    ):
        """
        Initialize JSON handler with optional immediate save/load.
        
        Args:
            filepath: Path to JSON file
            data: Data to save (if mode is 'write', 'append', or 'update')
            mode: Operation mode for initial save
            autoload: Automatically load data if file exists
            kwargs: Additional save parameters (indent, ensure_ascii, etc.)
        """
        self.filepath = Path(filepath) if filepath else None
        
        if self.filepath and self.filepath.suffix.lower() != '.json':
            self.filepath = self.filepath / 'file.json'
        
        if data is not None and self.filepath:
            print(f'save data to {filepath}')
            self.save(data, mode=mode, **kwargs)
        
        if data is None and self.filepath:
            print(f'read data from {filepath}')
            self.load()
        
        if not self.filepath and data is not None:
            print(f'init class with data and filepath is None')
            self.data = data 

    @staticmethod
    def _validate_element(element: Any) -> Any:
        """Internal method to validate and convert elements."""
        if isinstance(element, (str, int, float, bool)) or element is None:
            return element
        elif isinstance(element, np.ndarray):
            return element.tolist()
        elif isinstance(element, np.generic):
            return element.item()
        elif isinstance(element, (list, tuple)):
            return [JSON_Handler._validate_element(e) for e in element]
        elif isinstance(element, dict):
            return {k: JSON_Handler._validate_element(v) for k, v in element.items()}
        elif isinstance(element, datetime):
            return element.isoformat()
        elif hasattr(element, '__dict__'):
            return JSON_Handler._validate_element(element.__dict__)
        else:
            return str(element)

    @staticmethod
    def _deep_update(target: Dict, update: Dict) -> Dict:
        """Recursively update nested dictionaries."""
        for k, v in update.items():
            if isinstance(v, Mapping):
                target[k] = JSON_Handler._deep_update(target.get(k, {}), v)
            else:
                target[k] = v
        return target

    def save(
        self,
        data: Optional[dict | list] = None,
        filepath: Optional[Union[str, Path]] = None,
        mode: Literal['write', 'append', 'update'] = 'write',
        indent: int = 4,
        ensure_ascii: bool = False
    ) -> None:
        """
        Save data to JSON file with specified mode.
        
        Args:
            data: Data to save (uses instance data if None)
            filepath: Target file path (uses instance filepath if None)
            mode: How to handle existing data:
                - 'write': Overwrite completely
                - 'append': Shallow merge (top-level keys only)
                - 'update': Deep merge (preserves nested structures)
            indent: JSON indentation
            ensure_ascii: Force ASCII output
        """
        filepath = Path(filepath) if filepath else self.filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if not filepath:
            raise ValueError("No filepath provided")
           
        if filepath.suffix.lower() != '.json':
            filepath = filepath.with_suffix('.json')
            
        data = self._validate_element(data if data is not None else self.data)
            
        existing_data = {}
        if mode in ('append', 'update') and filepath.exists():
            existing_data = self.load(filepath)
            
            if not isinstance(existing_data, dict):
                existing_data = {}
                
            if mode == 'append':
                data = {**existing_data, **data} if isinstance(data, dict) else data
            elif mode == 'update':
                data = self._deep_update(existing_data, data) if isinstance(data, dict) else data
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            
        self.filepath = filepath
        self.data = data

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data = None

    def load(
        self,
        filepath: Optional[str | Path] = None
    ) -> dict | list:
        """
        Load data from JSON file.
        
        Args:
            filepath: Path to load from (uses instance filepath if None)
            update_instance: Whether to update the instance's data
            
        Returns:
            Parsed JSON data
        """
        if not filepath:
            filepath = self.filepath
        
        if isinstance(filepath, str):
            filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        if not filepath:
            raise ValueError("No filepath provided")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.data['data'] = np.asarray(self.data['data'])

    def update(
        self,
        updates: Union[Dict, List],
        filepath: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> None:
        """
        Convenience method for update mode save.
        """
        self.save(data=updates, filepath=filepath, mode='update', **kwargs)
    

class YAML_Config:
    def __init__(self, filepath: str = None, data: dict = None):
        self.filepath = Path(filepath) if filepath else None
        self.data = None

        if filepath is not None and data is None:
            self.data = self.load(filepath)
        elif data is not None:
            self.data = data
            if filepath is not None:
                self.save(filepath)

    def _convert_none(self, val):
        """Convert Python None to string 'None'."""
        return "None" if val is None else val

    def _convert_params(self, d):
        """Recursively convert None values in nested structures."""
        if isinstance(d, dict):
            return {k: self._convert_params(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._convert_params(v) for v in d]
        else:
            return self._convert_none(d)

    def _restore_none_and_literal(self, val):
        """Restore None and evaluate stringified lists/dicts."""
        if val == "None":
            return None
        elif isinstance(val, str):
            try:
                # Safely evaluate stringified lists/dicts/numbers/etc.
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return val
        return val

    def _recursive_restore(self, d):
        """Recursively restore None and literal values."""
        if isinstance(d, dict):
            return {k: self._recursive_restore(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._recursive_restore(v) for v in d]
        else:
            return self._restore_none_and_literal(d)

    def save(self, filepath: str = None):
        """Save config to YAML file after converting None to 'None'."""
        import yaml

        filepath = Path(filepath) if filepath else self.filepath
        if not filepath:
            raise ValueError("No filepath provided")

        cleaned_data = self._convert_params(self.data)

        with open(filepath, 'w') as f:
            yaml.dump(cleaned_data, f, sort_keys=False, default_flow_style=False, indent=4)

    def load(self, filepath: str = None):
        """Load and parse YAML file, restoring None and stringified literals."""
        import yaml

        filepath = Path(filepath) if filepath else self.filepath
        if not filepath:
            raise ValueError("No filepath provided")

        with open(filepath, 'r') as f:
            raw_data = yaml.safe_load(f)

        self.data = self._recursive_restore(raw_data)
        return self.data


def load_yaml_file(path_to_file:Path):
    import yaml

    with open(path_to_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_config_file(config_name: Path) -> dict:
    """
    Загружает конфигурационный файл из указанной директории.
    
    Args:
        config_name: Имя конфигурационного файла (c расширением .yaml)
        config_dir: Директория с конфигурационными файлами
        
    Returns:
        Словарь с конфигурацией
    """
    config_path = Path(config_name)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл {config_path} не найден")
    
    return load_yaml_file(config_path)


def save_config_file(config: dict, config_path: str):
    """Сохраняет конфигурацию в YAML-файл."""
    import yaml

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

