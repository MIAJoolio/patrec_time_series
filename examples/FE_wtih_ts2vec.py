"""

"""
from pathlib import Path

from patrec.feature_extraction import ts2vec_extract_features, visualize_latent_space,visualize_3_latent_space
from patrec.clustering import BIRCH_model, DBSCAN_model, AC_model,Sklearn_kmeans_model

def get_embeddings():
    train_repr, train_labels, test_repr, test_labels = ts2vec_extract_features(dataset_name= DATASET_NAME, train_val_ratio=TRAIN_VAL_SPLIT, model_config=MODEL_CONFIG) 
    
    return train_repr, train_labels, test_repr, test_labels
    
def main():
    global DATASET_NAME, TRAIN_VAL_SPLIT, MODEL_CONFIG 
     
    DATASET_NAME = "Car"
    for MODEL_CONFIG in ['configs/fe_default/ts2vec_config0.yaml', 'configs/fe_default/ts2vec_config1.yaml', 'configs/fe_default/ts2vec_config2.yaml', 'configs/fe_default/ts2vec_config3.yaml']:

        for split in [0.4,0.5,0.6,0.7,0.8]:
            TRAIN_VAL_SPLIT = split
            train_repr, train_labels, test_repr, test_labels = get_embeddings()
            
            Path(f'experiments/FE_with_ts2vec/{Path(MODEL_CONFIG).parts[-1].split('.')[0]}/res/').mkdir(exist_ok=True, parents=True)
            
            # visualize_latent_space(train_repr, train_labels, test_repr, test_labels, f'experiments/FE_with_ts2vec/latent_{DATASET_NAME}_{MODEL_CONFIG.split(".")[0][-1]}_{TRAIN_VAL_SPLIT}.png')
            visualize_3_latent_space(train_repr, train_labels, test_repr, test_labels, f'experiments/FE_with_ts2vec/{Path(MODEL_CONFIG).parts[-1].split('.')[0]}/pics/latent_{DATASET_NAME}_{"_".join(str(TRAIN_VAL_SPLIT).split("."))}.png')

            
            # Создаем модель BIRCH
            birch_model = BIRCH_model()
            # Загружаем YAML-конфигурацию и выполняем search_method
            df_birch_grid = birch_model.evaluate_results(X=test_repr, y_true=test_labels, config_path="configs/parameters_search/birch.yaml", search_method="grid_search")
            df_birch_grid.to_excel(f'experiments/FE_with_ts2vec/{Path(MODEL_CONFIG).parts[-1].split('.')[0]}/res/BIRCH_{DATASET_NAME}_{"_".join(str(TRAIN_VAL_SPLIT).split("."))}.xlsx')
            
            # Создаем модель DBSCAN
            dbscan_model = DBSCAN_model()
            # Загружаем YAML-конфигурацию и выполняем search_method
            df_dbscan_grid = dbscan_model.evaluate_results(X=test_repr, y_true=test_labels, config_path="configs/parameters_search/dbscan.yaml", search_method="grid_search")
            df_dbscan_grid.to_excel(f'experiments/FE_with_ts2vec/{Path(MODEL_CONFIG).parts[-1].split('.')[0]}/res/DBSCAN_{DATASET_NAME}_{"_".join(str(TRAIN_VAL_SPLIT).split("."))}.xlsx')
            
            # Создаем модель
            aggl_model = AC_model()
            # Автоматический подбор гиперпараметров
            df_aggl = aggl_model.evaluate_results(X=test_repr, y_true=test_labels, config_path="configs/parameters_search/agglomerative.yaml", search_method="grid_search")
            df_aggl.to_excel(f'experiments/FE_with_ts2vec/{Path(MODEL_CONFIG).parts[-1].split('.')[0]}/res/Aggl_clustering_{DATASET_NAME}_{"_".join(str(TRAIN_VAL_SPLIT).split("."))}.xlsx')
            
            # Создаем модель
            km_model = Sklearn_kmeans_model()
            # Автоматический подбор гиперпараметров
            km_grid = km_model.evaluate_results(X=test_repr, y_true=test_labels, config_path="configs/parameters_search/sklearn_kmeans.yaml", search_method="grid_search")
            km_grid.to_excel(f'experiments/FE_with_ts2vec/{Path(MODEL_CONFIG).parts[-1].split('.')[0]}/res/KMeans_{DATASET_NAME}_{"_".join(str(TRAIN_VAL_SPLIT).split("."))}.xlsx')
            
if __name__ == '__main__':
    main()