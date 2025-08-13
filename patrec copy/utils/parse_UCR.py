import os
import yaml
import pandas as pd
import numpy as np 

from bs4 import BeautifulSoup

from aeon.datasets import load_classification


__all__ = [
    'download_table_content',
    'download_UCR_dataset',
    'load_UCR'
]

def download_table_content(html_file_path, save_path):
    """
    Парсит таблицу с датасетами из локального HTML-файла и сохраняет в CSV
    """
    # Читаем HTML из файла
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Парсим с помощью BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Находим tbody (или tr напрямую)
    tbody = soup.find('tbody')
    if not tbody:
        tbody = soup  # если tbody отсутствует

    rows = []
    for tr in tbody.find_all('tr'):
        tds = [td.get_text(strip=True) for td in tr.find_all('td')]
        if tds:  # игнорируем пустые строки
            rows.append(tds)

    # Заголовки колонок
    headers = ['Dataset', 'Train Size', 'Test Size', 'Length', 'No. of Classes', 'Type']

    # Создаём DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Сохраняем в CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Таблица успешно сохранена в {save_path}")
    print(f"Всего записей: {len(df)}")
    return df


def download_UCR_dataset(dataset_name, save_dir="data/UCR_benchmark/datasets"):
    """
    Загружает датасет и сохраняет его с учётом длины временных рядов.
    Сохраняет YAML-файл с метаданными (length_type, task_type)
    :param dataset_name: Название датасета (например, 'ItalyPowerDemand')
    :param save_dir: Папка для сохранения данных
    :return: None
    """
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Загрузка данных
    try:
        X, y = load_classification(dataset_name)
    except Exception as e:
        print(f"Ошибка при загрузке датасета {dataset_name}: {e}")
        return

    metadata = {}

    # Определение length_type
    if isinstance(X, np.ndarray):
        print(f"{dataset_name} — Размеры X: {X.shape}, y: {y.shape}")
        if X.ndim == 3:
            # Все ряды одинаковой длины: (n_instances, n_channels, n_timepoints)
            is_equal_length = True
        else:
            # Не поддерживаемый формат
            print(f"Неподдерживаемый формат X для {dataset_name}: {type(X)}, ndim={X.ndim}")
            return
        
    elif isinstance(X, list):
        # Проверяем, одинакова ли длина всех временных рядов
        lengths = [ts.shape[-1] for ts in X if isinstance(ts, np.ndarray)]
        is_equal_length = False
    else:
        print(f"Неизвестный тип данных для {dataset_name}: {type(X)}")
        return

    metadata["length_type"] = "equal_length" if is_equal_length else "diff_length"

    # Определение task_type
    unique_classes = np.unique(y)
    metadata["task_type"] = "binary" if len(unique_classes) == 2 else "multi"

    # Сохраняем метаданные в YAML
    yaml_path = os.path.join(dataset_dir, "metadata.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    print(f"Metadata сохранен: {yaml_path}")

    # Сохранение в зависимости от типа
    if is_equal_length:
        # Сохраняем как один .npy файл
        x_save_path = os.path.join(dataset_dir, "raw_features.npy")
        np.save(x_save_path, X)
        print(f"raw_features.npy сохранён: {x_save_path}")
    else:
        # Сохраняем каждый временной ряд в отдельном .npy файле
        series_dir = os.path.join(dataset_dir, "raw_features")
        os.makedirs(series_dir, exist_ok=True)

        for i, series in enumerate(X):
            series_path = os.path.join(series_dir, f"{i}.npy")
            np.save(series_path, series)
        print(f"Временные ряды сохранены в папку: {series_dir}")

    # Сохраняем метки
    y_save_path = os.path.join(dataset_dir, "labels.csv")
    if isinstance(y, np.ndarray):
        np.savetxt(y_save_path, y, fmt='%s', delimiter=',')
    else:
        # Если y не numpy array, конвертируем в него
        np.savetxt(y_save_path, np.array(y), fmt='%s', delimiter=',')
    print(f"labels.csv сохранён: {y_save_path}")
    

def load_UCR(dataset_name, data_dir="data/UCR_benchmark/datasets", train_ratio=0.8):
    """
    Загружает временные ряды и метки из подготовленной структуры.
    Делит данные на train/test в зависимости от train_ratio.

    :param dataset_name: Название датасета, например 'Worms'
    :param data_dir: Путь к директории с датасетами
    :param train_ratio: Пропорция train-выборки (например, 0.8 = 80% train)
    :return: X_train, X_test, y_train, y_test, metadata
    """
    dataset_dir = os.path.join(data_dir, dataset_name)

    # Проверка существования директории
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Директория для датасета {dataset_name} не найдена: {dataset_dir}")

    # Загрузка метаданных
    metadata_path = os.path.join(dataset_dir, "metadata.yaml")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Файл metadata.yaml не найден в {dataset_dir}")
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)

    # Определяем типы
    length_type = metadata.get("length_type", "equal_length")

    # Загрузка меток
    labels_path = os.path.join(dataset_dir, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Файл labels.csv не найден в {dataset_dir}")
    y = pd.read_csv(labels_path, header=None).values

    # Загрузка временных рядов
    if length_type == "equal_length":
        features_path = os.path.join(dataset_dir, "raw_features.npy")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Файл raw_features.npy не найден в {dataset_dir}")
        X = np.load(features_path)
    else:
        features_dir = os.path.join(dataset_dir, "raw_features")
        if not os.path.exists(features_dir):
            raise FileNotFoundError(f"Папка raw_features не найдена в {dataset_dir}")

        # Сортировка файлов по номеру
        feature_files = sorted(
            [f for f in os.listdir(features_dir) if f.endswith(".npy")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        X = [np.load(os.path.join(features_dir, f)) for f in feature_files]

    # Проверка совпадения длин
    if len(X) != len(y):
        raise ValueError(f"Количество временных рядов ({len(X)}) не совпадает с количеством меток ({len(y)})")

    return X, y, metadata


def main():
    html_file_path = 'data/UCR_benchmark/table_of_content_raw.txt'
    save_path = 'data/UCR_benchmark/UCR_content_table.csv'

    table_of_content = download_table_content(html_file_path, save_path)
    print(table_of_content.head(), table_of_content.Type.unique())
    
    # исключены 'IMAGE', 'HAR', 'OTHER', 'EOG', 'EPG', 'MEG', 'EMG', 'HEMODYNAMICS', 'ECG','EEG',
    
    table_of_content = table_of_content.loc[table_of_content.Type.isin(['AUDIO',  'DEVICE', 'MOTION', 'SPECTRO', 'SIMULATED', 'SENSOR' 'TRAFFIC', 'FINANCIAL' 'SPEECH', 'HAR'])]
    print(table_of_content.shape, table_of_content.head(), table_of_content.Type.unique(), table_of_content['No. of Classes'].unique())
    
    # for dt_name in table_of_content['Dataset'].values:
    #     download_UCR_dataset(dt_name)
    download_UCR_dataset('UrbanSound') # MelbournePedestrian
    # download_UCR_dataset('Chinatown') # ItalyPowerDemand
    # download_UCR_dataset('ECG200')
    
    
if __name__ == '__main__':
    main()