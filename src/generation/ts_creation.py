import inspect
import hashlib
import json
import numpy as np

from typing import List, Dict, Optional, Tuple, Literal, Union, Callable
from dataclasses import dataclass
from pathlib import Path
from itertools import product

from src.utils import load_config_file, save_config_file, generate_combinations


all = [
    'Base_generator',
    'TS_generator',
    'Noise_generator',
    'TS_component',
    'TS_block',
    'Block_generator',
    'Block_merger'
]


class Base_generator:
    """
    Base class for time series generators. Caches the function signature for reuse.
    Attributes:
        gen_func (Callable): The time series generation function.
        _sig (inspect.Signature): Cached signature of the function.
    """
    def __init__(self, gen_func: Callable):
        self.gen_func = gen_func
        self._sig = inspect.signature(gen_func)

    def get_signature(self):
        """
        Returns the cached signature of the generator function.
        Returns:
            inspect.Signature: Signature of the generator function.
        """
        return self._sig


class TS_generator(Base_generator):
    """
    Generator for time series data based on a function signature or configuration file.
    Supports three modes:
    1. Generate parameters from function and save config.
    2. Use existing config to generate time series.
    3. Extend with new generator functions using compatible config/dict format.
    Attributes:
        gen_func (Callable): Function used to generate time series.
        config (dict): Configuration dictionary containing parameter bounds and settings.
    """
    def __init__(self, gen_func: Callable, config: Union[dict, str] = None):
        """
        Initialize the time series generator.
        Args:
            gen_func (Callable): A callable that generates a time series.
            config (Union[dict, str], optional): Configuration dict or path to config file. Defaults to None.
        """
        super().__init__(gen_func)
        self.config = {}
        if isinstance(config, dict):
            self.config = config.copy()
        elif isinstance(config, str):
            try:
                self.config = load_config_file(config)
            except Exception as e:
                raise ValueError(f"Failed to load config file: {e}")
        if 'random_state' in self.config:
            np.random.seed(self.config['random_state'])

    def generate_params(self, round_val: int = 3, all_values: bool = False, random_state: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Generate parameter values based on function signature or loaded config.
        Args:
            round_val (int): Number of decimal places for generated values.
            all_values (bool): If True, returns all possible values instead of a random choice.
            random_state (Optional[int]): Seed for reproducibility.
        Returns:
            dict: Dictionary of generated parameter values.
        """
        if random_state is not None:
            np.random.seed(random_state)
        elif 'random_state' in self.config:
            np.random.seed(self.config['random_state'])

        params = {}
        length = self.config.get('length', None)

        for name, param in self._sig.parameters.items():
            if name == "length":
                length = self.config.get(name, param.default)
            else:
                d = self.config.get(f"{name}_d", -1.0 if param.default is inspect.Parameter.empty else param.default)
                u = self.config.get(f"{name}_u", 1.0 if d >= 0 else abs(d))
                q = self.config.get(f"{name}_q", 10)

                values = np.linspace(d, u, q).round(round_val)
                value = values if all_values else np.random.choice(values)
                params[name] = value

        if random_state is not None:
            self.config['random_state'] = random_state

        params['length'] = length
        self.config['length'] = length

        structured_params = {
            'general': {
                'length': length,
                'type': self.__class__.__name__,
                'generator_name': self.gen_func.__name__
            },
            'gen_params': params
        }

        return structured_params

    def generate(self, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate time series using generated or provided parameters.
        Args:
            **kwargs: Arguments passed to `generate_params`.
        Returns:
            Union[np.ndarray, List[np.ndarray]]: Generated time series or list of series.
        """
        if kwargs.get('all_values'):
            params_grid = self.generate_params(**kwargs)
            params_list = []
            keys = list(params_grid['gen_params'].keys()) + ['length']
            vals = [params_grid['gen_params'][key] for key in keys if isinstance(params_grid['gen_params'][key], np.ndarray)] + [params_grid['general']['length']]

            index_product = product(*[range(len(v)) for v in vals])
            for indices in index_product:
                param = {key: vals[i][indices[i]] for i, key in enumerate(keys)}
                params_list.append(param)

            return [self.gen_func(**param) for param in params_list], params_list
        params = self.generate_params(**kwargs)
        return self.gen_func(**params['gen_params']), params

    def save_config(self, config_path: str = 'configs/config_file.yaml') -> str:
        """
        Save current configuration to file with length-based naming.
        Args:
            config_path (str): Path to config file.
        Returns:
            str: Full path to saved config file.
        """
        path = Path(config_path)
        base_name = path.stem
        ext = path.suffix
        full_name = f"{base_name}_{self.config['length']}{ext}"
        save_config_file(self.config, full_name)
        return full_name


class Noise_generator(Base_generator):
    """
    Generator for adding noise to time series data.
    
    Allows flexible initialization without requiring raw data upfront.
    Can generate noise of specified length or match input data shape at runtime.
    
    Attributes:
        gen_func (Callable): Function that generates noise.
        config (dict): Configuration dictionary with parameter bounds and settings.
        default_length (Optional[int]): Default length if no data is provided later.
    """
    def __init__(self, gen_func: Callable, config: Union[dict, str] = None, data:Optional[np.ndarray] = None, length: Optional[int] = None):
        """
        Initialize the noise generator without requiring input data.
        
        Args:
            gen_func (Callable): A callable that generates noise of given length.
            config (Union[dict, str], optional): Configuration dict or path to config file.
            default_length (Optional[int]): Default length to use if no data is passed later.
        """
        super().__init__(gen_func)
        self.config = {}

        if isinstance(config, dict):
            self.config = config.copy()
        elif isinstance(config, str):
            try:
                self.config = load_config_file(config)
            except Exception as e:
                raise ValueError(f"Failed to load config file: {e}")

        if data is not None:
            self.data = data
            self.length = data.shape[0]
        if data is None:
            self.length = length
            self.data = np.arange(self.length)
        
        if 'random_state' in self.config:
            np.random.seed(self.config['random_state'])

    def generate_params(self, round_val: int = 3, all_values: bool = False, random_state: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Generate noise parameters based on function signature or loaded config.
        
        Args:
            round_val (int): Number of decimal places for generated values.
            all_values (bool): If True, returns all possible values instead of a random choice.
            random_state (Optional[int]): Seed for reproducibility.
            
        Returns:
            dict: Dictionary of generated parameter values.
        """
        if random_state is not None:
            np.random.seed(random_state)
        elif 'random_state' in self.config:
            np.random.seed(self.config['random_state'])

        params = {}

        for name, param in self._sig.parameters.items():
            if name == "data":
                continue 
            else:
                d = self.config.get(f"{name}_d", 0 if param.default is inspect.Parameter.empty else param.default)
                u = self.config.get(f"{name}_u", 1.0 if d >= 0 else abs(d))
                q = self.config.get(f"{name}_q", 10)

                values = np.linspace(d, u, q).round(round_val)
                value = values if all_values else np.random.choice(values)
                params[name] = value

        if random_state is not None:
            self.config['random_state'] = random_state

        structured_params = {
            'general': {
                'length': self.length,
                'type': self.__class__.__name__,
                'generator_name': self.gen_func.__name__
            },
            'gen_params': params
        }

        return structured_params

    def generate(self, data:Optional[np.ndarray]=None, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Apply noise to the provided data or generate standalone noise of specified length.
        
        Args:
            data (Optional[np.ndarray]): Data to apply noise to. If None, uses default_length.
            **kwargs: Arguments passed to `generate_params`.
            
        Returns:
            tuple[np.ndarray, dict]: Noised data and the parameters used.
        """
        params = self.generate_params(**kwargs)
        if data is None:
            data = self.data
        noise = self.gen_func(data, **params['gen_params'])

        return noise, params

    def save_config(self, config_path: str = 'configs/config_file.yaml') -> str:
        """
        Save current configuration to file with shape-based naming.
        
        Args:
            config_path (str): Path to config file.
            
        Returns:
            str: Full path to saved config file.
        """
        path = Path(config_path)
        base_name = path.stem
        ext = path.suffix

        full_name = f"{base_name}_{self.default_length if self.default_length else 'unknown'}{ext}"
        safe_config = {k: v for k, v in self.config.items() if k != 'data'}
        save_config_file(safe_config, full_name)
        return full_name


class TS_component:
    """
    Represents a single component of a time series (e.g., trend, wave, shift, noise).
    Attributes:
        data (np.ndarray): The actual values of the component.
        params (dict): Parameters used to generate this component.
        component_type (str): Type of the component ('trend', 'wave', 'shift', 'noise').
        hash (str): MD5 hash of the sorted parameters for reproducibility tracking.
    """
    def __init__(self, data: np.ndarray, params: dict = None, component_type: str = 'unknown'):
        self.data = data
        self.params = params or {'general': {'length': len(data)}, 'gen_params': {}}
        if not self.params.get('length'):
            self.params['general']['length'] = len(data)
        self.component_type = component_type
        self.hash = self._hash_params()

    def _hash_params(self) -> str:
        """
        Generate an MD5 hash from the sorted parameter dictionary.
        Returns:
            str: Hex digest of the MD5 hash.
        """
        if not self.params:
            return ""
        param_str = json.dumps(self.params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()


class TS_Block:
    """
    A block representing a time series composed of multiple components.
    Components can be added via add_component method and accessed via .components.
    """
    def __init__(self):
        self.components: Dict[str, TS_component] = {}

    def add_component(self, component: TS_component):
        """
        Add a new component to the block.
        Args:
            component (TS_component): Component to add.
        """
        self.components[component.component_type] = component

    def get_series(self) -> np.ndarray:
        """
        Get the full time series by summing all components.
        Returns:
            np.ndarray: Combined time series.
        """
        return sum((comp.data for comp in self.components.values()), 
                   np.zeros_like(next(iter(self.components.values())).data))

    def get_metadata(self) -> dict:
        """
        Get metadata of all components including their hashes and parameters.
        Returns:
            dict: Metadata dictionary with component types as keys.
        """
        return {
            comp_type: {
                "hash": comp.hash,
                "params": comp.params
            }
            for comp_type, comp in self.components.items()
        }

    def has_component(self, component_type: str) -> bool:
        """
        Check if a component of given type is present.
        Args:
            component_type (str): Component type to check.
        Returns:
            bool: True if component exists.
        """
        return component_type in self.components


class Block_generator:
    """
    Generates a TS_Block from provided generators for each component.
    Supports generating trend, wave (seasonality), shift, and noise components.
    """
    def __init__(self):
        pass

    def generate_block(
        self,
        trend_gen: Optional['TS_generator'] = None,
        wave_gen: Optional['TS_generator'] = None,
        shift_gen: Optional['TS_generator'] = None,
        noise_gen: Optional['Noise_generator'] = None
    ) -> TS_Block:
        """
        Generate a TS_Block using provided generators for each component.
        Args:
            trend_gen (TS_generator, optional): Generator for trend component.
            wave_gen (TS_generator, optional): Generator for wave (seasonality) component.
            shift_gen (TS_generator, optional): Generator for structural shifts.
            noise_gen (Noise_generator, optional): Generator for noise.
            length (int): Default length if no generator provides it.
        Returns:
            TS_Block: Generated time series block.
        """
        block = TS_Block()

        # Автоопределение длины
        for gen in [trend_gen, wave_gen, shift_gen, noise_gen]:
            if gen is not None:
                if hasattr(gen, 'config') and 'length' in gen.config:
                    length = gen.config['length']
                elif hasattr(gen, 'data'):
                    length = gen.data.shape[0]
                break

        components = {}

        if trend_gen:
            data, params = trend_gen.generate()
            components['trend'] = (data, params)

        if wave_gen:
            data, params = wave_gen.generate()
            components['wave'] = (data, params)

        if shift_gen:
            data, params = shift_gen.generate()
            components['shift'] = (data, params)

        base_signal = np.zeros(length)
        for name, (data, _) in components.items():
            if len(data) != length:
                raise ValueError(f"Component '{name}' has unexpected length: {len(data)} vs expected {length}")
            base_signal += data

        if noise_gen:
            noise_gen = Noise_generator(noise_gen.gen_func, noise_gen.config, base_signal)
            noise_data, noise_params = noise_gen.generate()
            block.add_component(TS_component(noise_data, noise_params, 'noise'))

        for name, (data, params) in components.items():
            block.add_component(TS_component(data, params, name))

        return block


class Block_merger:
    """
    Manages multiple TS_Blocks and merges them into a single time series.
    Keeps track of components and metadata during merging.
    """
    def __init__(self):
        self.blocks: List[TS_Block] = []

    def add_block(self, block: TS_Block):
        """
        Add a block to the merger.
        Args:
            block (TS_Block): Block to add.
        """
        self.blocks.append(block)

    def remove_block(self, index: int):
        """
        Remove a block at a specific index.
        Args:
            index (int): Index of block to remove.
        """
        if 0 <= index < len(self.blocks):
            self.blocks.pop(index)
        else:
            raise IndexError("Block index out of range")

    def merge(self, smooth_transition: bool = True) -> Dict[str, Union[np.ndarray, List]]:
        """
        Merge all blocks into one continuous time series.
        Args:
            smooth_transition (bool): Whether to smooth transition between blocks.
        Returns:
            dict: Dictionary containing merged series, components, and metadata.
        """
        full_series = None
        components_data = {key: [] for key in ['trend', 'wave', 'shift', 'noise']}

        for block in self.blocks:
            current_components = block.components
            for key in components_data:
                if key in current_components:
                    components_data[key].append(current_components[key].data)
                else:
                    components_data[key].append(np.zeros_like(full_series if full_series is not None else np.zeros(1)))

            block_series = block.get_series()
            if full_series is None:
                full_series = block_series.copy()
            else:
                if smooth_transition:
                    start_point = block_series[0]
                    end_point = full_series[-1]
                    block_series = block_series - start_point + end_point
                full_series = np.concatenate([full_series, block_series[1:]])

        return {
            "series": full_series,
            "components": components_data,
            "metadata": [block.get_metadata() for block in self.blocks]
        }


def main():
    from src.utils.visuals import plot_series
    from ts_generators import linear_trend, sin_wave
    
    real_data = TS_component(np.array([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]), {}, 'real_data')

    linear_config = load_config_file('src/generation/base_configs/trend/basic_linear_trend.yaml')
    # linear_config['slope_q'] = 1 
    
    trend_gen = TS_generator(linear_trend, linear_config)
    seasonal_gen = TS_generator(sin_wave, 'src/generation/base_configs/wave/basic_sin_wave.yaml')

    bg = Block_generator()
    block1 = bg.generate_block(trend_gen=trend_gen, wave_gen=seasonal_gen)
    block2 = bg.generate_block(trend_gen=trend_gen, wave_gen=seasonal_gen)
    block3 = bg.generate_block(trend_gen=trend_gen, wave_gen=seasonal_gen)
    
    real_block = TS_Block()
    real_block.add_component(real_data)
    
    merger = Block_merger()
    merger.add_block(block1)
    merger.add_block(block2)
    merger.add_block(block3)
    merger.add_block(real_block)
    result = merger.merge()

    print("Series shape:", result["series"].shape)
    print(result["metadata"])
    plot_series(result['series'], save_path='output2.png')
    plot_series([block.get_series() for block in [block1, block2, block3, real_block]], save_path='output_all.png')
    
if __name__ == '__main__':
    main()