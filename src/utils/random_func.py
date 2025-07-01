from itertools import product

import numpy as np


def generate_combinations(param_ranges: dict) -> list:
    """
    Генерирует все возможные комбинации параметров из словаря.
    
    :param param_ranges: Словарь, где ключи — названия параметров, значения — списки/массивы допустимых значений.
    :return: Список словарей с уникальными комбинациями параметров.
    """
    keys = param_ranges.keys()
    value_lists = [item if isinstance(item,np.ndarray) else [item] for item in param_ranges.values()]
    combinations = []

    for values in product(*value_lists):
        combinations.append(dict(zip(keys, values)))

    return combinations
