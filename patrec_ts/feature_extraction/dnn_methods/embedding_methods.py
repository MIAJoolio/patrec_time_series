from typing import Dict, Any, Optional
import sys
import os
import time

import numpy as np

from ts2vec import TS2Vec

cost_path = os.path.abspath('patrec/feature_extraction/CoST')
if cost_path not in sys.path:
    sys.path.insert(0, cost_path)

from patrec_ts.feature_extraction.CoST.cost import CoST
from patrec_ts.feature_extraction.CoST.models.encoder import CoSTEncoder
from patrec_ts.utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan

from patrec_ts.feature_extraction.fe_classes import BaseExtractor, FE_result


class TS2VecExtractor(BaseExtractor):
    """Экстрактор признаков с использованием TS2Vec."""
    
    def __init__(
            self,
            config: dict[str, Any],
            pretrained_path: Optional[str] = None,
            train_on_init: bool = False,
            input_data: Optional[np.ndarray] = None
    ) -> None:
        """
        Инициализация экстрактора TS2Vec.
        
        Args:
            config: Конфигурация модели и обучения
            pretrained_path: Путь к предобученной модели (если None - обучение с нуля)
            train_on_init: Обучить модель сразу при инициализации
            input_data: Данные для обучения (если train_on_init=True)
        """
        self.config = config
        self.pretrained_path = pretrained_path
        self.model = None
        self._initialize_model()
        
        if train_on_init and input_data is not None:
            self.train_model(input_data)
    
    def _initialize_model(self) -> None:
        """Инициализация модели TS2Vec."""
        model_params = self.config['model_init']
        
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            self.model = TS2Vec.load(self.pretrained_path)
        else:
            self.model = TS2Vec(
                input_dims=1,  
                output_dims=model_params['output_dims'],
                hidden_dims=model_params['hidden_dims'],
                depth=model_params['depth'],
                lr=model_params.get('lr', 0.001),
                device=model_params.get('device', 'cuda')
            )
    
    def train_model(self, data: np.ndarray) -> None:
        """Обучение модели на предоставленных данных."""
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)  # (n_samples, seq_len, 1)
            
        train_params = self.config['model_training']
        
        self.model.fit(
            data,
            n_epochs=train_params['n_epochs'],
            n_iters=train_params.get('n_iters'),
            verbose=train_params['verbose']
        )
        
        if 'model_save_path' in self.config:
            os.makedirs(os.path.dirname(self.config['model_save_path']), exist_ok=True)
            self.model.save(self.config['model_save_path'])
    
    def extract(self, data: np.ndarray, **kwargs) -> FE_result:
        """
        Извлечение признаков с помощью TS2Vec.
        
        Args:
            data: Входные данные временного ряда (n_samples, seq_len)
            **kwargs: Дополнительные параметры
            
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
                        
        Returns:
            FE_result: Результат извлечения признаков
        """
        start_time = time.time()
        
        # Подготовка данных
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)  # (n_samples, seq_len, 1)
        
        # Параметры кодирования
        encode_params = self.config.get('encode_params', {})
        encoding_window = encode_params.get('encoding_window', 'full_series')
        
        # Извлечение признаков
        if encoding_window == 'full_series':
            repr = self.model.encode(data, 
                                     mask=encode_params.get('mask', None), encoding_window='full_series')
        else:
            repr = self.model.encode(
                data,
                mask=encode_params.get('mask', None), 
                encoding_window=encoding_window,
                sliding_length=encode_params.get('sliding_length', None),
                sliding_padding=encode_params.get('sliding_padding', None)
            )
        
        
        return FE_result(
            component=None,
            method_name='cost_embedding',
            method_params={
                'config': self.config
            },
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape,
                'output_shape': repr.shape
            },
            results={
                'embedding': repr
            }
        )
        
        
class CoSTExtractor(BaseExtractor):
    """Экстрактор признаков на основе CoST (Contrastive Shapelet and Time)."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 pretrained_path: Optional[str] = None,
                 train_on_init: bool = False,
                 input_data: Optional[np.ndarray] = None):
        """
        Args:
            config: Конфигурация модели
            pretrained_path: Путь к предобученной модели
            train_on_init: Обучить сразу при инициализации
            input_data: Данные для обучения
        """
                
        self.config = config
        self.pretrained_path = pretrained_path
        self.model = None
        
        # Инициализация модели
        self._initialize_model()
        
        if train_on_init and input_data is not None:
            self.train_model(input_data)
    
    def _initialize_model(self):
        """Инициализация модели CoST."""
        model_params = self.config['model_init']
        
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            self.model = CoST(
                input_dims=model_params.get('input_dim',1) ,  # Временной ряд одномерный
                kernels=model_params.get('kernels', [1, 2, 4, 8, 16, 32, 64, 128]),
                alpha=model_params.get('alpha', 0.0005),
                max_train_length=model_params.get('max_train_length', 3000),
                device=model_params.get('device', 'cuda'),
                output_dims=model_params.get('output_dims', 320),
                hidden_dims=model_params.get('hidden_dims', 64),
                lr=model_params.get('lr', 0.001),
                batch_size=model_params.get('batch_size', 8)
            )
            self.model.load(self.pretrained_path)
        else:
            self.model = CoST(
                input_dims=1,
                kernels=model_params.get('kernels', [1, 2, 4, 8, 16, 32, 64, 128]),
                alpha=model_params.get('alpha', 0.0005),
                max_train_length=model_params.get('max_train_length', 3000),
                device=model_params.get('device', 'cuda'),
                output_dims=model_params.get('output_dims', 320),
                hidden_dims=model_params.get('hidden_dims', 64),
                lr=model_params.get('lr', 0.001),
                batch_size=model_params.get('batch_size', 8)
            )

    def train_model(self, data: np.ndarray):
        """Обучение модели CoST."""
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)  # (n_samples, seq_len, 1)
            
        train_params = self.config['model_training']
        
        loss_log = self.model.fit(
            data,
            n_epochs=train_params.get('n_epochs', 20),
            n_iters=train_params.get('n_iters'),
            verbose=train_params.get('verbose', True)
        )
        
        if 'model_save_path' in self.config:
            os.makedirs(os.path.dirname(self.config['model_save_path']), exist_ok=True)
            self.model.save(self.config['model_save_path'])
            
        return loss_log

    def extract(self, data: np.ndarray, **kwargs) -> FE_result:
        """Извлечение признаков с помощью CoST."""
        start_time = time.time()
        
        # Подготовка данных
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)  # (n_samples, seq_len, 1)
        
        # Параметры кодирования
        encode_params = self.config.get('encode_params', {})
        encoding_window = encode_params.get('encoding_window', 'full_series')
        
        # Извлечение признаков
        repr = self.model.encode(
            data,
            mode='forecasting',
            encoding_window=encoding_window,
            sliding_length=encode_params.get('sliding_length', None),
            sliding_padding=encode_params.get('sliding_padding', 0),
            batch_size=encode_params.get('batch_size', None)
        )
        
        return FE_result(
            component=None,
            method_name='cost_embedding',
            method_params={
                'config': self.config
            },
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape,
                'output_shape': repr.shape
            },
            results={
                'embedding': repr
            }
        )

    def save_model(self, path: str):
        """Сохранение модели."""
        self.model.save(path)