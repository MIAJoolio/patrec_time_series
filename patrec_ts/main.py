from pathlib import Path
import time

import numpy as np
import pandas as pd

from aeon.clustering import (
    TimeSeriesKMeans as aeonTimeSeriesKMeans
    , TimeSeriesKShape
    , TimeSeriesCLARA
    , KSpectralCentroid
    )
from tslearn.clustering import TimeSeriesKMeans as TSLearnTimeSeriesKMeans
from kshape.core import KShapeClusteringCPU

from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from patrec_ts.preprocessing.manual_methods.parametric.trend_linear_preprocessor import TrendLinearPreprocessor
from patrec_ts.preprocessing.manual_methods.season.dft_preprocessor import SeasonDFTPreprocessor
from patrec_ts.utils import download_UCR_dataset
from patrec_ts.utils.files_helper import JSON_Handler, load_yaml_file
from patrec_ts.utils.loaders import StratifiedTSLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import warnings


def load_data(ds_name:str):
    """
    
    """
    
    download_UCR_dataset(ds_name, )

    ds_path = Path('data/UCR_benchmark/datasets') / ds_name
    JSON_Handler(
        filepath=ds_path / 'ds.json',
        data={
            'data': np.load(ds_path / 'raw_features.npy'),
            'labels': pd.read_csv(ds_path / 'labels.csv', header=None).iloc[:, 0].values
        },
        mode='write'
    )

    data: StratifiedTSLoader = StratifiedTSLoader(
        filepath=ds_path / 'ds.json',
    )
    model_encoder = LabelEncoder().fit(data.splits['train']['labels'])
    labels = model_encoder.transform(data.splits['train']['labels'])

    labels_test = LabelEncoder().fit_transform(data.splits['test']['labels'])

    return data.splits['train']['data'], labels, data.splits['test']['data'], labels_test


def save_results(
    results_df: pd.DataFrame,
    save_dir: Path,
    method_name: str,
    time_eval: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_pred: np.ndarray,
    cluster_centers: np.ndarray,
    config_file: str
):
    """
    Сохраняет все результаты: метрики, время, центроиды, конфиги.
    """
    # Создаём директорию для метода
    method_dir = Path(save_dir) / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем таблицу метрик (уже в DataFrame)
    results_df.to_csv(method_dir / "metrics.csv", index=False)

    # Сохраняем время обучения и инференса
    time_df = pd.DataFrame([{
        'method': method_name,
        'train_time_sec': time_eval['train'],
        'infer_time_sec': time_eval['infer']
    }])
    time_df.to_csv(method_dir / "timing.csv", index=False)

    # Сохраняем центроиды (если есть)
    if cluster_centers is not None:
        np.save(method_dir / "centroids.npy", cluster_centers)

    # Сохраняем метки предсказаний
    np.save(method_dir / "y_pred_train.npy", y_pred)
    np.save(method_dir / "y_true_train.npy", y_train)

    # Сохраняем конфиг
    config_path = Path('.configs/benchmark') / config_file
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_content = f.read()
        with open(method_dir / "config.yaml", 'w') as f:
            f.write(config_content)

    # Сохраняем полные метрики (включая ARI, NMI, RI)
    metrics_summary = {
        'method': method_name,
        'ARI_train': adjusted_rand_score(y_train, y_pred),
        'RI_train': rand_score(y_train, y_pred),
        'NMI_train': normalized_mutual_info_score(y_train, y_pred),
        'ARI_test': adjusted_rand_score(y_train, y_pred), 
        'train_time_sec': time_eval['train'],
        'infer_time_sec': time_eval['infer'],
        'n_clusters': len(np.unique(y_pred)),
        'n_series': len(y_train),
        'is_multivariate': X_train.ndim == 3
    }
    
    # Сохраняем summary в JSON
    with open(method_dir / "summary.json", 'w') as f:
        import json
        json.dump(metrics_summary, f, indent=4)

    print(f"[Save] Результаты для {method_name} сохранены в {method_dir}")

def visualize(
    X_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_centers: np.ndarray,
    method_name: str,
    save_dir: Path,
    is_multivariate: bool = False
):
    """
    Визуализирует результаты кластеризации: 
    - Центроиды кластеров
    - Примеры из каждого кластера
    - Матрицу ошибок (TP/FP/TN/FN) при сравнении с истинными метками
    - График распределения кластеров
    """
    # # Убедимся, что данные корректны
    # if X_train.ndim == 2:
    #     n_series, length = X_train.shape
    #     n_features = 1
    # elif X_train.ndim == 3:
    #     n_series, length, n_features = X_train.shape
    # else:
    #     raise ValueError("X_train должен быть 2D или 3D массивом")

    # unique_labels = np.unique(y_true)
    # n_clusters = len(unique_labels)

    # # Определим, есть ли переобучение (например, если число кластеров > числа уникальных меток)
    # predicted_labels = np.unique(y_pred)
    # n_pred_clusters = len(predicted_labels)

    # # Создаём папку для графиков
    # viz_dir = Path(save_dir) / "visualizations" / method_name
    # viz_dir.mkdir(parents=True, exist_ok=True)

    # # --- 1. Визуализация центроидов (если они есть) ---
    # if cluster_centers is not None:
    #     centers = cluster_centers
    #     if centers.ndim == 2:
    #         centers = centers.reshape(centers.shape[0], 1, centers.shape[1])
    #     elif centers.ndim == 3:
    #         pass
    #     else:
    #         raise ValueError("cluster_centers должны быть 2D или 3D")

    #     fig, axes = plt.subplots(nrows=1, ncols=min(n_pred_clusters, 5), figsize=(18, 4))
    #     if n_pred_clusters == 1:
    #         axes = [axes]
    #     for i in range(min(n_pred_clusters, 5)):
    #         ax = axes[i]
    #         if is_multivariate:
    #             for j in range(centers.shape[2]):
    #                 ax.plot(centers[i, 0, :], label=f'Feature {j+1}')
    #         else:
    #             ax.plot(centers[i, 0, :], color='black', linewidth=2)
    #         ax.set_title(f'Centroid Cluster {i} (n={len(y_pred[y_pred==i])})')
    #         ax.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(viz_dir / "centroids.png", dpi=150)
    #     plt.close()

    # # --- 2. Визуализация примеров по кластерам ---
    # # Сгруппируем данные по кластерам
    # cluster_examples = {}
    # for cid in np.unique(y_pred):
    #     idxs = np.where(y_pred == cid)[0]
    #     if len(idxs) > 0:
    #         cluster_examples[cid] = X_train[idxs]

    # # Выберем до 3 примеров на кластер
    # max_examples_per_cluster = 3
    # fig, axes = plt.subplots(nrows=n_pred_clusters, ncols=max_examples_per_cluster, figsize=(16, 4*n_pred_clusters))
    # if n_pred_clusters == 1:
    #     axes = [axes]
    # elif max_examples_per_cluster == 1:
    #     axes = axes.reshape(-1, 1)

    # for i, cid in enumerate(cluster_examples.keys()):
    #     examples = cluster_examples[cid]
    #     selected = examples[:max_examples_per_cluster]
    #     for j, ex in enumerate(selected):
    #         ax = axes[i][j]
    #         if is_multivariate:
    #             for k in range(ex.shape[1]):
    #                 ax.plot(ex[:, k], label=f'F{k+1}')
    #         else:
    #             ax.plot(ex, color='blue')
    #         ax.set_title(f"Cluster {cid}, Sample {j+1}")
    #         ax.grid(True)
    # plt.suptitle(f"Examples per Cluster ({method_name})", fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(viz_dir / "examples_per_cluster.png", dpi=150)
    # plt.close()

    # # --- 3. Визуализация матрицы ошибок (confusion matrix) ---
    # # Для этого нужно сопоставить кластеры с истинными метками
    # # Используем greedy assignment: найдём лучшее соответствие между кластерами и истинными классами
    # from sklearn.metrics.cluster import adjusted_rand_score

    # # Попробуем найти оптимальное соответствие между кластерами и true labels
    # # Это важно, потому что кластеры могут быть перенумерованы произвольно
    # def find_best_assignment(y_true, y_pred):
    #     unique_true = np.unique(y_true)
    #     unique_pred = np.unique(y_pred)
    #     n_true = len(unique_true)
    #     n_pred = len(unique_pred)

    #     # Создадим матрицу совпадений
    #     match_matrix = np.zeros((n_true, n_pred))

    #     for t_idx, t_label in enumerate(unique_true):
    #         for p_idx, p_label in enumerate(unique_pred):
    #             match_count = np.sum((y_true == t_label) & (y_pred == p_label))
    #             match_matrix[t_idx, p_idx] = match_count

    #     # Найдём оптимальное соответствие (assignment)
    #     from scipy.optimize import linear_sum_assignment
    #     row_ind, col_ind = linear_sum_assignment(-match_matrix)  # минимизация потерь
    #     mapping = {unique_pred[col]: unique_true[row] for row, col in zip(row_ind, col_ind)}
    #     return mapping

    # try:
    #     best_mapping = find_best_assignment(y_true, y_pred)
    #     y_pred_mapped = np.array([best_mapping[label] for label in y_pred])
    # except Exception as e:
    #     print(f"Ошибка при сопоставлении меток: {e}")
    #     y_pred_mapped = y_pred  # fallback

    # # Теперь строим confusion matrix
    # cm = confusion_matrix(y_true, y_pred_mapped, labels=np.unique(y_true))
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    # plt.title(f"Confusion Matrix (Mapped Labels)\n{method_name}")
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.tight_layout()
    # plt.savefig(viz_dir / "confusion_matrix.png", dpi=150)
    # plt.close()

    # # --- 4. Интерактивная визуализация с Plotly: scatter plot по первым двум компонентам PCA ---
    # # Используем PCA для снижения размерности (даже для multivariate — можно развернуть)
    
    # # Преобразуем временные ряды в плоские векторы
    # if X_train.ndim == 3:
    #     X_flat = X_train.reshape(X_train.shape[0], -1)  # (n, length * features)
    # else:
    #     X_flat = X_train

    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_flat)

    # df_plot = pd.DataFrame({
    #     'x': X_pca[:, 0],
    #     'y': X_pca[:, 1],
    #     'true_label': y_true.astype(str),
    #     'pred_label': y_pred_mapped.astype(str),
    #     'cluster_id': y_pred
    # })

    # fig = px.scatter(df_plot, x='x', y='y',
    #                  color='true_label',
    #                  hover_data=['pred_label', 'cluster_id'],
    #                  title=f"PCA Projection of Time Series (True vs Pred) - {method_name}",
    #                  labels={'x': 'PC1', 'y': 'PC2'},
    #                  opacity=0.7)
    # fig.update_layout(showlegend=True)
    # fig.write_html(viz_dir / "pca_interactive.html")
    # fig.write_image(viz_dir / "pca_interactive.png", width=800, height=600)

    viz_dir = ''
    print(f"[Visualize] Визуализация для {method_name} сохранена в {viz_dir}")
    
    
def make_benchmark(dataset_name: str, save_dir: Path):
    """
    Запускает бенчмаркинг всех алгоритмов с визуализацией и сохранением.
    """
    
    mapping = {
        'aeon_TimeSeriesKMeans.yaml': aeonTimeSeriesKMeans,
        'aeon_TimeSeriesKMeans_dtw.yaml': aeonTimeSeriesKMeans,
        'aeon_TimeSeriesCLARA.yaml': TimeSeriesCLARA,  
        # 'aeon_TimeSeriesKShape.yaml': TimeSeriesKShape,
        # 'aeon_KSpectralCentroid.yaml': KSpectralCentroid,
        
        'TSLearnTimeSeriesKMeans.yaml':TSLearnTimeSeriesKMeans,
        'TSLearnTimeSeriesKMeans_dtw.yaml': TSLearnTimeSeriesKMeans,
        
        'KshapeKShapeClusteringCPU.yaml':KShapeClusteringCPU, 
    }

    X_train, y_train, X_test, y_test = load_data(dataset_name)
    time_eval = {}
    print(X_train.shape)

    # Предварительная проверка типа данных
    is_multivariate = X_train.ndim == 3

    # Создаём общий DataFrame для хранения всех результатов
    all_results = []

    for k, v in mapping.items():
        print(f"\n=== Running benchmark for {k} ===")
        
        if k == 'KshapeKShapeClusteringCPU.yaml':
            
            train_dim = X_train.shape
            X_train = X_train.reshape(train_dim[0], train_dim[2], train_dim[1])
            test_dim = X_test.shape
            X_test = X_test.reshape(test_dim[0], test_dim[2], test_dim[1])

        pipeline = Pipeline(steps=[
            ('cluster_method', v(**load_yaml_file(Path('.configs/benchmark') / k))),
        ])

        train_time_st = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time_en = time.perf_counter()

        infer_time_st = time.perf_counter()
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        infer_time_en = time.perf_counter()

        # Получаем центроиды (если доступны)
        try:
            centers = pipeline.named_steps['cluster_method'].cluster_centers_
        except AttributeError:
            centers = None

        # Оценка метрик
        ari_train = np.round(adjusted_rand_score(y_train, y_pred_train), 5)
        ri_train = np.round(rand_score(y_train, y_pred_train), 5)
        nmi_train = np.round(normalized_mutual_info_score(y_train, y_pred_train), 5)

        ari_test = np.round(adjusted_rand_score(y_test, y_pred_test), 5)
        ri_test = np.round(rand_score(y_test, y_pred_test), 5)
        nmi_test = np.round(normalized_mutual_info_score(y_test, y_pred_test), 5)

        # Сохраняем метрики в список
        result_row = {
            'method': k,
            'ARI_train': ari_train,
            'RI_train': ri_train,
            'NMI_train': nmi_train,
            'ARI_test': ari_test,
            'RI_test': ri_test,
            'NMI_test': nmi_test,
            'train_time_sec': train_time_en - train_time_st,
            'infer_time_sec': infer_time_en - infer_time_st,
            'n_clusters': len(np.unique(y_pred_train)),
            'dataset': dataset_name,
            'is_multivariate': is_multivariate
        }
        all_results.append(result_row)

        # Вывод в консоль
        print(
            'TRAIN: RI: ', ri_train,
            ' ARI: ', ari_train,
            ' NMI: ', nmi_train,
        )
        print(
            'TEST: RI: ', ri_test,
            ' ARI: ', ari_test,
            ' NMI: ', nmi_test,
        )

        # Визуализация
        try:
            visualize(
                X_train=X_train,
                y_true=y_train,
                y_pred=y_pred_train,
                cluster_centers=centers,
                method_name=k,
                save_dir=save_dir,
                is_multivariate=is_multivariate
            )
        except Exception as e:
            print(f"[Error] Визуализация для {k} не удалась: {e}")

        # Сохранение результатов
        save_results(
            results_df=pd.DataFrame([result_row]),
            save_dir=save_dir,
            method_name=k,
            time_eval={'train': train_time_en - train_time_st, 'infer': infer_time_en - infer_time_st},
            X_train=X_train,
            y_train=y_train,
            y_pred=y_pred_train,
            cluster_centers=centers,
            config_file=k
        )

        time_eval[k] = {'train': train_time_en - train_time_st, 'infer': infer_time_en - infer_time_st}

    # Сохраняем общую таблицу всех результатов
    final_results_df = pd.DataFrame(all_results)
    final_results_df.to_csv(Path(save_dir) / "benchmark_overall.csv", index=False)
    print(f"\n[INFO] Все результаты сохранены в {save_dir}/benchmark_overall.csv")

    return final_results_df

if __name__ == '__main__':

    make_benchmark('BasicMotions','result/first_bench')
    # make_benchmark('MelbournePedestrian','')


# ds_name = 'BasicMotions' 
# # ds_name = 'MelbournePedestrian'


# pipeline = Pipeline(steps=[
#     # ('trend', TrendLinearPreprocessor()),
#     # ('season', SeasonDFTPreprocessor()),
#     # ('cluster_method', TimeSeriesCLARA(n_clusters=11)),
#     ('cluster_method', TimeSeriesKMeans(n_clusters=10, distance='euclidean')),
#     # ('cluster_method', TSLearnTimeSeriesKMeans(n_clusters=4))
# ])


# pipeline.fit(data.splits['train']['data'], labels)
# res = pipeline.predict(data.splits['train']['data'])

# print(
#     'TRAIN: RI: ', rand_score(labels, res),
#     ' ARI: ', adjusted_rand_score(labels, res),
#     ' NMI: ', normalized_mutual_info_score(labels, res),
# )
# res_test = pipeline.predict(data.splits['test']['data'])

# print(
#     'TEST: RI: ', rand_score(labels_test, res_test),
#     ' ARI: ', adjusted_rand_score(labels_test, res_test),
#     ' NMI: ', normalized_mutual_info_score(labels_test, res_test),
# )

# # DataSet MelbournePedestrian
# # 1. Scores only with ClusterMethods (train + test)
# # 2. Разные комбинация детрейна, десезенности, дешум и депики и тд
# # 3. Это все в ml.flow (в jupyter ноутбуке)

# print(1)