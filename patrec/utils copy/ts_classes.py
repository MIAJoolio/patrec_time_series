from typing import List, Dict, Optional, Tuple, Literal, Union, Callable, Any
from abc import ABC, abstractmethod

from enum import Enum, auto
from pathlib import Path
import json
import hashlib

import numpy as np

all = [
    'TS_component_type',
    'TS_component',
    'TS_block',
    'TS_additive_block',
    'TS_multiplicative_block',
    'Block_connector'
]


class TS_component_type(Enum):
    """Enumeration of possible time series component types"""
    SIGNAL = auto()
    TREND = auto()
    SEASONAL = auto()
    SHIFT = auto()
    NOISE = auto()
    RESIDUAL = auto()


class TS_component:
    """
    Atomic component of a time series with enhanced functionality.
    
    Attributes:
        data (np.ndarray): Component values
        params (dict): Generation parameters
        component_type (TS_component_type): Type of component
        hash (str): Unique hash of component parameters
        name (str): Optional human-readable name
    """
    def __init__(self, 
                 data: np.ndarray, 
                 component_type: TS_component_type,
                 params: Optional[dict] = None,
                 name: Optional[str] = None):
        self.data = np.asarray(data)
        self.component_type = component_type
        self.params = params or {}
        self.name = name or f"{component_type.name.lower()}_component"
        self.hash = self._generate_hash()
        
        # Validate length consistency in parameters
        if 'length' in self.params and self.params['length'] != len(self.data):
            raise ValueError("Parameter length doesn't match data length")

    def _generate_hash(self) -> str:
        """Generate reproducible hash from component parameters"""
        hash_data = {
            'type': self.component_type.name,
            'params': self.params,
            'length': len(self.data)
        }
        return hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()

    def __add__(self, other: Union['TS_component', np.ndarray, float]) -> np.ndarray:
        """Enable component arithmetic operations"""
        if isinstance(other, TS_component):
            return self.data + other.data
        return self.data + other

    def __len__(self) -> int:
        return len(self.data)


class TS_block(ABC):
    """
    Abstract base class for time series blocks with common functionality.
    """
    def __init__(self):
        self._components: Dict[TS_component_type, TS_component] = {}
        self._metadata: Dict[str, Any] = {}

    def __len__(self) -> int:
        """Returns length of the time series block (determined by its components)"""
        if not self._components:
            return 0
        # All components should have same length (enforced in add_component)
        return len(next(iter(self._components.values())))

    @property
    def components(self) -> Dict[TS_component_type, TS_component]:
        return self._components

    @property
    def length(self) -> int:
        """Alias for __len__ for explicit access"""
        return len(self)
    
    def add_component(self, component: TS_component):
        """Add a component with type checking and length validation"""
        if not isinstance(component, TS_component):
            raise TypeError("Component must be TS_component instance")
            
        if self._components and len(component) != self.length:
            raise ValueError("All components must have same length")
            
        self._components[component.component_type] = component

    def get_component(self, component_type: TS_component_type) -> Optional[TS_component]:
        return self._components.get(component_type)

    def has_component(self, component_type: TS_component_type) -> bool:
        return component_type in self._components

    @abstractmethod
    def build_series(self) -> np.ndarray:
        """Build the complete time series from components"""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata about all components"""
        return {
            'components': {
                comp_type.name: {
                    'hash': comp.hash,
                    'params': comp.params,
                    'name': comp.name
                }
                for comp_type, comp in self._components.items()
            },
            'block_metadata': self._metadata
        }


class TS_additive_block(TS_block):
    """
    Time series block where components are summed together.
    """
    def build_series(self) -> np.ndarray:
        if not self._components:
            return np.array([])
            
        base = np.zeros(self.length)
        for component in self._components.values():
            base += component.data
        return base


class TS_multiplicative_block(TS_block):
    """
    Time series block where components are multiplied together.
    """
    def build_series(self) -> np.ndarray:
        if not self._components:
            return np.array([])
            
        base = np.ones(self.length)
        for component in self._components.values():
            base *= component.data
        return base


class Block_connector:
    """
    Advanced connector for time series blocks with multiple manipulation capabilities.
    """
    def __init__(self):
        self._blocks: List[TS_block] = []
        self._transition_cache: Dict[int, np.ndarray] = {}

    def add_block(self, block: TS_block, index: Optional[int] = None):
        """Add block at specific position or append"""
        if index is None:
            self._blocks.append(block)
        else:
            self._blocks.insert(index, block)

    def remove_block(self, index: int):
        """Remove block by index"""
        if 0 <= index < len(self._blocks):
            self._blocks.pop(index)
            # Clear transition cache for this position
            self._transition_cache.pop(index, None)

    def get_blocks(self) -> List[TS_block]:
        return self._blocks.copy()

    def build_series(self, 
                    smooth_transitions: bool = True,
                    transition_window: int = 10) -> np.ndarray:
        """
        Build complete series with optional smooth transitions between blocks.
        
        Args:
            smooth_transitions: Whether to smooth transitions between blocks
            transition_window: Number of points to use for smoothing
        """
        if not self._blocks:
            return np.array([])

        full_series = self._blocks[0].build_series()
        
        for i, block in enumerate(self._blocks[1:], 1):
            block_series = block.build_series()
            
            if smooth_transitions and len(full_series) > 0:
                # Calculate or get cached transition
                if i not in self._transition_cache:
                    transition = self._create_transition(
                        full_series[-transition_window:],
                        block_series[:transition_window],
                        transition_window
                    )
                    self._transition_cache[i] = transition
                else:
                    transition = self._transition_cache[i]
                
                # Connect with transition
                full_series = np.concatenate([
                    full_series[:-transition_window],
                    transition,
                    block_series[transition_window:]
                ])
            else:
                full_series = np.concatenate([full_series, block_series])
                
        return full_series

    def _create_transition(self, 
                         end_segment: np.ndarray, 
                         start_segment: np.ndarray,
                         window: int) -> np.ndarray:
        """Create smooth transition between two segments"""
        if len(end_segment) != window or len(start_segment) != window:
            raise ValueError("Transition segments must match window size")
            
        weights = np.linspace(1, 0, window)
        return end_segment * weights + start_segment * (1 - weights)

    def get_component_series(self, component_type: TS_component_type) -> np.ndarray:
        """Get concatenated series of specific component type from all blocks"""
        series = []
        for block in self._blocks:
            if comp := block.get_component(component_type):
                series.append(comp.data)
            else:
                series.append(np.zeros(block.length))
        return np.concatenate(series) if series else np.array([])

    def rearrange_blocks(self, new_order: List[int]):
        """Reorder blocks according to provided indices"""
        if len(new_order) != len(self._blocks):
            raise ValueError("New order must include all blocks")
        self._blocks = [self._blocks[i] for i in new_order]
        self._transition_cache = {}  # Clear transition cache

    def get_metadata(self) -> Dict[str, Any]:
        """Get combined metadata from all blocks"""
        return {
            'blocks': [block.get_metadata() for block in self._blocks],
            'total_length': sum(len(block) for block in self._blocks)
        }