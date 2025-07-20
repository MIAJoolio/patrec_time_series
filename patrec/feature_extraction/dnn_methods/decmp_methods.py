from typing import Dict, Any, Optional
import sys
import os
import time

import numpy as np
import pandas as pd

from patrec.feature_extraction.old_fe_classes import Base_extractor, FE_result
from neuralprophet import NeuralProphet, set_log_level

class NeuralProphet_extractor(Base_extractor):
    """Экстрактор компонент временного ряда с использованием NeuralProphet"""
    
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
        self.freq = self.config.get('freq', 'D')
        self.model = None
        self._initialize_model()
        
        if train_on_init and input_data is not None:
            self.train_model(input_data)

    def _initialize_model(self) -> None:
        """Безопасная инициализация модели"""
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            try:
                # Попытка стандартной загрузки
                self.model = NeuralProphet.load(self.pretrained_path)
            except Exception as e:
                # Fallback для PyTorch 2.6+
                try:
                    import torch
                    from neuralprophet.configure import ConfigSeasonality
                    
                    with torch.serialization.safe_globals([ConfigSeasonality]):
                        self.model = NeuralProphet.load(self.pretrained_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to load model: {str(e)}")
        else:
            model_params = {k: v for k, v in self.config['model_params'].items() 
                          if k not in ['n_lags', 'n_forecasts']}
            self.model = NeuralProphet(**model_params)
    
    def _prepare_data(self, data: np.ndarray) -> pd.DataFrame:
        """Подготовка DataFrame для NeuralProphet"""
        dfs = []
        for i, ts in enumerate(data):
            dfs.append(pd.DataFrame({
                'ds': pd.date_range(start='2000-01-01', periods=len(ts)), 
                'y': ts,
                'ID': f'series_{i}'
            }))
        return pd.concat(dfs)

    def train_model(self, data: np.ndarray) -> None:
        """Обучение модели"""
        if len(data.shape) != 2:
            raise ValueError("Input shape should be (n_samples, n_timestamps)")
            
        train_df = self._prepare_data(data)
        self.model.fit(train_df, freq=self.freq, **self.config['train_params'])
        
        if 'model_save_path' in self.config:
            os.makedirs(os.path.dirname(self.config['model_save_path']), exist_ok=True)
            self.model.save(self.config['model_save_path'])

    def extract(self, data: np.ndarray, **kwargs) -> FE_result:
        """
        Декомпозиция ряда на компоненты без прогнозирования
        
        Args:
            data: массив формы (n_samples, n_timestamps)
            
        Returns:
            FE_result с разложенными компонентами
        """
        start_time = time.time()
        
        if len(data.shape) != 2:
            raise ValueError("Ожидается форма (n_samples, n_timestamps)")
        
        components = {
            'trend': [],
            'seasonal': [],
            'residuals': []
        }
        forecast_dfs = []
        
        for ts in data:
            df = pd.DataFrame({
                'ds': pd.date_range(start='2000-01-01', periods=len(ts)),
                'y': ts
            })
            
            # Получаем компоненты для исторических данных
            future = self.model.make_future_dataframe(df, periods=0)
            forecast = self.model.predict(future)
            decomp = self.model.predict_components(future)
            
            # Сохраняем компоненты
            components['trend'].append(decomp['trend'].values)
            components['seasonal'].append(
                decomp.drop(columns=['ds', 'trend']).sum(axis=1).values
            )
            components['residuals'].append(ts - decomp.drop(columns=['ds']).sum(axis=1).values)
            forecast_dfs.append(forecast)
        
        execution_time = time.time() - start_time
        
        return FE_result(
            component=None,  # Основной компонент не возвращаем, так как разложено на части
            method_name='NeuralProphet_Decomposition',
            method_params=self.config,
            execution_stats={
                'execution_time': execution_time,
                'input_shape': data.shape
            },
            results={
                'components': {
                    'trend': np.array(components['trend']),
                    'seasonal': np.array(components['seasonal']),
                    'residuals': np.array(components['residuals'])
                },
                'forecast_dfs': forecast_dfs,
                'model': self.model
            }
        )

# class NeuralProphet_extractor(Base_extractor):
#     """Экстрактор компонент временного ряда с использованием NeuralProphet"""
    
#     def __init__(self, 
#                  freq: str = 'D',
#                  n_lags: int = 0,
#                  n_forecasts: int = 1,
#                  trend_reg: float = 0,
#                  seasonality_mode: str = 'additive',
#                  yearly_seasonality: bool = True,
#                  weekly_seasonality: bool = True,
#                  daily_seasonality: bool = True,
#                  epochs: int = 10,
#                  batch_size: int = 32):
#         """
#         Инициализация NeuralProphet экстрактора.
        
#         Args:
#             freq: Частота данных ('D' - daily, 'H' - hourly и т.д.)
#             n_lags: Количество лагов для автокорреляции
#             n_forecasts: Количество шагов прогноза
#             trend_reg: Регуляризация тренда
#             seasonality_mode: 'additive' или 'multiplicative'
#             yearly_seasonality: Включать годовую сезонность
#             weekly_seasonality: Включать недельную сезонность
#             daily_seasonality: Включать дневную сезонность
#             epochs: Количество эпох обучения
#             batch_size: Размер батча
#         """
#         self.freq = freq
#         self.n_lags = n_lags
#         self.n_forecasts = n_forecasts
#         self.trend_reg = trend_reg
#         self.seasonality_mode = seasonality_mode
#         self.yearly_seasonality = yearly_seasonality
#         self.weekly_seasonality = weekly_seasonality
#         self.daily_seasonality = daily_seasonality
#         self.epochs = epochs
#         self.batch_size = batch_size
        
#         # Инициализируем модель
#         self.model = NeuralProphet(
#             n_lags=n_lags,
#             n_forecasts=n_forecasts,
#             trend_reg=trend_reg,
#             seasonality_mode=seasonality_mode,
#             yearly_seasonality=yearly_seasonality,
#             weekly_seasonality=weekly_seasonality,
#             daily_seasonality=daily_seasonality
#         )
        
#         # Настройка логирования
#         logging.getLogger('neuralprophet').setLevel(logging.WARNING)
    
#     def extract(self, data: np.ndarray, **kwargs) -> FE_result:
#         """
#         Извлекает компоненты временного ряда с помощью NeuralProphet.
        
#         Args:
#             data: Входной временной ряд
#             **kwargs: Дополнительные параметры
            
#         Returns:
#             FE_result: Результат с выделенными компонентами
#         """
#         start_time = time.time()
        
#         # Создаем DataFrame для NeuralProphet
#         df = pd.DataFrame({
#             'ds': pd.date_range(start='2000-01-01', periods=len(data), freq=self.freq),
#             'y': data
#         })
        
#         try:
#             # Обучаем модель
#             metrics = self.model.fit(df, freq=self.freq, epochs=self.epochs, batch_size=self.batch_size)
            
#             # Прогнозируем для получения компонент
#             future = self.model.make_future_dataframe(df, periods=0, n_historic=len(data))
#             forecast = self.model.predict(future)
            
#             # Извлекаем компоненты
#             components = self.model.predict_components(future)
            
#             # Основной компонент - сумма всех компонент (тренд + сезонности)
#             combined_component = components.drop(columns=['ds']).sum(axis=1).values
            
#             execution_time = time.time() - start_time
            
#             return FE_result(
#                 component=combined_component,
#                 method_name='NeuralProphet',
#                 method_params={
#                     'freq': self.freq,
#                     'n_lags': self.n_lags,
#                     'n_forecasts': self.n_forecasts,
#                     'trend_reg': self.trend_reg,
#                     'seasonality_mode': self.seasonality_mode,
#                     'yearly_seasonality': self.yearly_seasonality,
#                     'weekly_seasonality': self.weekly_seasonality,
#                     'daily_seasonality': self.daily_seasonality,
#                     'epochs': self.epochs,
#                     'batch_size': self.batch_size
#                 },
#                 execution_stats={
#                     'execution_time_sec': execution_time,
#                     'input_shape': data.shape,
#                     'training_metrics': metrics.to_dict() if metrics is not None else None
#                 },
#                 results={
#                     'components_df': components,
#                     'forecast_df': forecast,
#                     'model': self.model,
#                     'residuals': data - combined_component,
#                     'component_breakdown': {
#                         col: components[col].values 
#                         for col in components.columns 
#                         if col != 'ds'
#                     }
#                 }
#             )
            
#         except Exception as e:
#             raise RuntimeError(f"NeuralProphet extraction failed: {str(e)}")