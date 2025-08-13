from typing import List, Dict, Optional, Tuple, Union, Literal
from pathlib import Path
import json

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.utils.data import Dataset, DataLoader, random_split

from patrec.utils import load_config_file, setup_logger, plot_series
from patrec.generation.ts_generators import Time_series_generators_catalog
from patrec.generation.ts_noise_generators import Noise_generators_catalog

# добавим logger
logger = setup_logger(__name__, level='debug')

__all__ = [
    "Basic_generator",
    "Basic_dataset",
    "save_generated_data",
    "split_train_test",
    "generate_synthetic_dataset"
]   


class Basic_generator:
    """
    Класс для генерации временного ряда из блоков.
    """
    def __init__(self, ts_catalog: Optional[Time_series_generators_catalog] = None, noise_catalog: Optional[Noise_generators_catalog] = None, config_path:Path=None
    ):
        # Инициализация каталогов генераторов
        self.ts_catalog = ts_catalog or Time_series_generators_catalog()
        logger.debug(f'Generators catalog was initialized automatically:\n{ts_catalog is None}')
        self.noise_catalog = noise_catalog or Noise_generators_catalog()
        logger.debug(f'Noise generators catalog was initialized automatically:\n{noise_catalog is None}')
        
        # Информация по блокам 
        # Список для хранения блоков
        self.blocks = []
        self.blocks_length = []
        
        # если параметры задаются через файл конфигурацию, то добавление блоков выполняется автоматически
        if config_path is not None:
            blocks = self._read_config_file(config_path)
            for block in blocks:
                self.add_block(**block)
        
    def add_block(self, block_length: int, ts_generator, ts_params=None, noise_generator=None, noise_params=None, random_state=None):
        """
        Добавление блока генерации. 
        
        1. Уточнение пары функций генерации и генерации параметров 
        2. Уточнение интервалов функций генерации параметров 
        3. Длина отрезка временного ряда   
        """
        ts_generator_type = ts_generator if isinstance(ts_generator, str) else None
        noise_generator_type = noise_generator if isinstance(noise_generator, str) else None
        
        # Проверяем параметры генерации и генератор
        ts_generator, ts_params = self._resolve_generator(ts_generator, ts_params, 'ts', random_state)
        # Аналогично, но для шума если он есть
        noise_generator, noise_params = self._resolve_generator(noise_generator, noise_params, 'noise', random_state)

        # Добавление блока
        self.blocks.append({
            'ts_generator': ts_generator,
            'ts_params': ts_params,
            'ts_generator_type':ts_generator_type,
            'noise_generator': noise_generator,
            'noise_params': noise_params,
            'noise_generator_type':noise_generator_type
        })
        self.blocks_length.append(block_length)

        logger.debug(f'Added block with parameters:\n{self.blocks[-1]}')

    def _read_config_file(self, config_path) -> List[dict]:
        """
        Загрузка конфигурации из YAML файла.
        """
        blocks = []
        
        config = load_config_file(config_path)
        logger.debug(f'Generators and parameters were initialized by config:\npath={config_path}\n{config}')
        
        for inx, block_config in enumerate(config):
            
            ts_generator = block_config.get('ts_generator')
            logger.debug(f"Generator type:\n{ts_generator}")
            ts_params = block_config.get('ts_params', None)
            logger.debug(f"Generator parameters:\n{ts_params}")
            noise_generator = block_config.get('noise_generator')
            logger.debug(f"Noise generator type:\n{noise_generator}")
            noise_params = block_config.get('noise_params', None)
            logger.debug(f"Noise generator parameters:\n{noise_params}")
            block_length = block_config.get('block_length', 100)
            logger.debug(f"Block length:\n{block_length}")
            random_state = block_config.get('random_state', None)
            logger.debug(f"Random state:\n{random_state}")

            # Получаем конфигурации для обновления параметров каталога
            ts_gen_params = block_config.get('ts_gen_params', None)
            logger.debug(f"New ts_catalog parameters:\n{ts_gen_params}")
            
            if ts_gen_params is not None:
                # создаем новый тип генератора на основе старого
                new_ts_generator_name = '_'.join([ts_generator,str(inx)])
                new_ts_generator = self.ts_catalog.get_generator(ts_generator)
                self.ts_catalog.add_generator(new_ts_generator_name, new_ts_generator['generator'], new_ts_generator['params_generator'])
                # переобозначаем обратно
                ts_generator = new_ts_generator_name
                self.ts_catalog.update_generator_params(ts_generator, ts_gen_params)
            
            noise_gen_params = block_config.get('noise_gen_params', None)
            logger.debug(f"New noise_catalog parameters:\n{noise_gen_params}")
            
            if noise_gen_params is not None:
                # создаем новый тип генератора на основе старого
                new_noise_generator_name = '_'.join([noise_generator,str(inx)])
                new_noise_generator = self.noise_catalog.get_generator(noise_generator)
                self.noise_catalog.add_generator(new_noise_generator_name, new_noise_generator['generator'], new_noise_generator['params_generator'])
                # переобозначаем обратно
                noise_generator = new_noise_generator_name
                self.noise_catalog.update_generator_params(noise_generator, noise_gen_params)

            blocks.append({
                'block_length':block_length,
                'ts_generator': ts_generator,
                'ts_params': ts_params,
                'noise_generator': noise_generator,
                'noise_params': noise_params,
                'random_state': random_state
            })
                        
        return blocks

    def remove_block(self, index: int):
        """
        Удаление блока по индексу.
        Args:
            index: Индекс блока для удаления.
        """
        if 0 <= index < len(self.blocks):
            removed_block = self.blocks.pop(index)
            print(f"Removed block at index {index}: {removed_block}")
            self.blocks_length.pop(index)
        else:
            raise IndexError(f"Invalid block index: {index}")

    def update_block_params(self, index: int, new_ts_params: Dict = None, new_noise_params: Dict = None):
        """
        Обновление параметров блока.
        Args:
            index: Индекс блока для обновления.
            new_ts_params: Новые параметры временного ряда.
            new_noise_params: Новые параметры шума.
        """
        if 0 <= index < len(self.blocks):
            if new_ts_params is not None:
                self.blocks[index]['ts_params'].update(new_ts_params)
                print(f"Updated block {index} time-series params: {new_ts_params}")
            if new_noise_params is not None:
                self.blocks[index]['noise_params'].update(new_noise_params)
                print(f"Updated block {index} noise params: {new_noise_params}")
        else:
            raise IndexError(f"Invalid block index: {index}")

    def generate(self, random_state=None, mode: Literal['determinated', 'random'] = "determinated") -> np.ndarray:
        """
        Генерация временного ряда.
        Args:
            mode: Режим генерации
        """
        if not self.blocks:
            raise ValueError("No blocks available for generation")
        
        time_series = np.array([], dtype=np.float64)
        end_pts = {}

        for idx, (block, block_length) in enumerate(zip(self.blocks, self.blocks_length)):
            ts_generator = block['ts_generator']
            ts_params = block['ts_params']
            noise_generator = block['noise_generator']
            noise_params = block['noise_params']

            if mode == 'random':
                ts_params = self._generate_random_params(block['ts_generator_type'], 'ts')
                logger.debug(f'TS random generation parameters:\n{ts_params}')
                noise_params = self._generate_random_params(block['noise_generator_type'], 'noise')
                logger.debug(f'Noise random generation parameters:\n{noise_params}')

            else:
                logger.debug(f'TS determinated generation parameters:\n{ts_params}')
                logger.debug(f'Noise determinated generation parameters:\n{noise_params}')


            # Проверяем, что параметры не None
            if ts_params is None:
                raise ValueError(f"Missing parameters for time series generator in block {idx}")
            
            ts_length = block_length if idx == 0 else block_length + 1
            block_series = ts_generator(**ts_params, length=ts_length)

            if idx > 0 and end_pts.get(idx - 1) is not None:
                block_series += end_pts[idx - 1]

            if noise_generator and noise_params:
                noise = noise_generator(data=block_series, random_state=random_state, **noise_params)
                block_series = block_series.astype(np.float64)
                block_series += noise

            time_series = np.concatenate([time_series, block_series[1:] if idx > 0 else block_series])
            end_pts[idx] = block_series[-1]

        return time_series

    def generate_multiple(self, num_series: int, mode: str = "random") -> List[np.ndarray]:
        """
        Генерация нескольких временных рядов.
        """
        series_list = []
        for _ in range(num_series):
            series = self.generate(mode=mode)
            series_list.append(series)
        return series_list

    def _resolve_generator(self, generator, params=None, catalog_type:Literal['ts','noise']='ts', random_state=None):
        
        # Т.к. функция используется для проверки генерации шума, то при его отсутствии сохраняем None значения
        if generator is None and params is None:
            logger.debug(f"Generator and parameters are not set!")
            return None, None
        
        catalog = self.ts_catalog if catalog_type == 'ts' else self.noise_catalog
        
        if isinstance(generator, str):
            logger.debug(f"Generator function is from catalog:\n{generator}")
            # Получаем информацию о генераторе из каталога
            generator_info = catalog.get_generator(generator)
            generator = generator_info['generator']
        
            # Если параметры не заданы, используем автоматическую генерацию
            if params is None:
                params = generator_info['params_generator'](random_state=random_state)
                logger.debug(f'Parameters generated automatically:\n{params}')
            
            else:
                logger.debug(f'Parameters were set: {params}')
            
        else:
            # Если generator является функцией
            logger.debug(f"Generator function is not from catalog:\n{generator}")
        
        return generator, params
    
    def _generate_random_params(self, gen_type:str, catalog_type:Literal['ts', 'noise']) -> dict:

        # Т.к. функция используется для проверки генерации шума, то при его отсутствии сохраняем None значения
        if gen_type is None:
            return None
        
        catalog = self.ts_catalog if catalog_type == 'ts' else self.noise_catalog
        
        return catalog.get_generator(gen_type)['params_generator']()


class Basic_dataset(Dataset):
    """
    Класс для создания датасета временных рядов.
    """
    def __init__(self, data_path: str, normalize: bool = False, norm_type: Literal['minmax', 'zscore', 'robust'] = "minmax"):
        self.data = self.load_data(data_path)
        self.normalize = normalize
        self.norm_type = norm_type
        
        if normalize:
            self.scaler = self._get_scaler(norm_type)
            
            # Собираем все ряды в один массив для обучения scaler
            all_series = np.vstack([s.reshape(-1, 1) if s.ndim == 1 else s for s in self.data["series"]])  # [total_sequence_length, input_dim]
            self.scaler.fit(all_series)
            
            # Применяем нормализацию к каждому ряду
            self.series = [self.normalize_series(series) for series in self.data["series"]]
        else:
            self.series = [series.reshape(series.shape[0],-1) for series in self.data["series"]]
        
        self.labels = self.data["labels"]
        
    def load_data(self, data_path: str) -> Dict[str, Union[List[np.ndarray], List[int]]]:
        """
        Загрузка данных из JSON файла.
        """
        data = load_config_file(data_path)
        series = [np.array(item['row']) for item in data]
        labels = [item['class_id'] for item in data]
        return {"series": series, "labels": labels}

    def _get_scaler(self, norm_type: str):
        """
        Возвращает объект нормализации.
        """
        if norm_type == "minmax":
            return MinMaxScaler()
        elif norm_type == "zscore":
            return StandardScaler()
        elif norm_type == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def normalize_series(self, series: np.ndarray) -> np.ndarray:
        """
        Нормализует временной ряд.
        
        Args:
            series: Временной ряд размерности [sequence_length, input_dim].
        
        Returns:
            np.ndarray: Нормализованный временной ряд.
        """
        if series.ndim == 1:
            series = series.reshape(-1, 1)  # Преобразуем одномерный ряд в двумерный
        
        # Проверяем, что размерность данных соответствует ожиданиям scaler
        if series.shape[1] != self.scaler.n_features_in_:
            raise ValueError(
                f"Input data has {series.shape[1]} features, but the scaler expects {self.scaler.n_features_in_} features."
            )
        
        normalized_series = self.scaler.transform(series)
        return normalized_series

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        """
        Возвращает временной ряд и метку.
        
        Returns:
            Tuple[np.ndarray, int]: Временной ряд размерности [sequence_length, input_dim] и метка.
        """
        series = self.series[idx]
        label = self.labels[idx]
        return series, label


def save_generated_data(data: List[Dict], save_path: str):
    """
    Сохранение данных в JSON файл.
    """
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=4)


def split_train_test(dataset: Dataset, train_ratio: float = 0.8, random_state: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Разделение датасета на train и test выборки с использованием random_split.
    Args:
        dataset: Исходный датасет (например, Time_series_dataset).
        train_ratio: Доля данных для обучающей выборки (по умолчанию 0.8).
        random_state: Случайное состояние для воспроизводимости.
    Returns:
        Кортеж с Dataset для train и test.
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    # Определяем размеры выборок
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Разбиваем датасет
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def generate_synthetic_dataset(config_paths:List[str], save_path:str='data/synth_dataset/dataset.json', num_series:int=100):
    """
    Функция для генерации toy-dataset синтетических данных 
    """
    
    full_data = []
    save_path = Path(save_path)
    config_paths = config_paths if isinstance(config_paths, list) else Path(config_paths).iterdir()
    
    for inx, path in enumerate(sorted(config_paths)):
        cluster_generator = Basic_generator(config_path=path)
        cluster_data = cluster_generator.generate_multiple(num_series=num_series)
        
        plot_series(cluster_data[:10], [i for i in range(10)], plot_title=f'Cluster {inx+1}', save_path=save_path.parent / f'cluster_{inx+1}.png')
        full_data += [{'class_id':inx+1, 'row':list(row)} for row in cluster_data]
    
    save_generated_data(full_data, save_path=save_path)