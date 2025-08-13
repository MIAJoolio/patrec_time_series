from typing import Dict, Any, Optional, Union
from pathlib import Path
import mlflow

from patrec_ts.utils.files_helper import JSON_Handler, YAML_Config


class MLflowExperimentLogger:
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "http://localhost:5000",
        tags: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        temp_path:Path = '.tmp/'
    ):
        """Initialize experiment logger with automatic run management.
        
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking server URI
            tags: Dictionary of tags for the run
            run_name: Optional name for this run
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self.run_name = run_name
        self.temp_path = Path(temp_path)
        
        # Configure MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        # Start run immediately on initialization
        self._start_run()

    def _start_run(self):
        """Start a new MLflow run with configured settings."""
        self.active_run = mlflow.start_run(run_name=self.run_name)
        for tag_name, tag_value in self.tags.items():
            mlflow.set_tag(tag_name, tag_value)

    def _end_run(self):
        """End the current MLflow run if active."""
        if hasattr(self, 'active_run') and self.active_run:
            mlflow.end_run()
            self.active_run = None

    def __del__(self):
        """Clean up by ending run when logger is destroyed."""
        self._end_run()

    def log_parameters(self, params: Union[Dict, str, Path]=None):
        """Log parameters ONLY as artifact in params/ folder.
        
        Args:
            params: Either dict of parameters or path to YAML/JSON file
        """
        if not params:
            params = self.temp_path/'params.yaml'

        if isinstance(params, (str, Path)):
            mlflow.log_artifact(params, artifact_path="params")
            params_dict = YAML_Config(params)
            mlflow.log_params(params_dict.data['data'])
            
        elif isinstance(params, dict):
            params_dict = JSON_Handler(filepath=self.temp_path / 'params.json', data=params)

            mlflow.log_artifact(params_dict.filepath, artifact_path="params")
            

    def log_metrics(self, metrics: Union[Dict, str, Path]=None):
        """Log metrics from dictionary or JSON file, handling both scalar and list values.
        
        Args:
            metrics: Either dict of metrics or path to JSON file
                    Can contain single values or lists of values
        """
        if not metrics:
            metrics = self.temp_path/'metrics.json'
        
        if isinstance(metrics, (Path, str)):
            mlflow.log_artifact(metrics, artifact_path="metrics")
            
            metrics_dict = JSON_Handler(metrics)
            
            # for k,v in metrics_dict.data['data'].items():
            mlflow.log_metrics(metrics_dict.data['data'].reshape(-1,1)[0][0])
            # if not isinstance(v, list):
            #     mlflow.log_param(k,v)
            # else:
            #     for step, item in enumerate(v):
                    
        
        elif isinstance(metrics, dict):
            metrics_dict = JSON_Handler(filepath=self.temp_path/'metrics.json', data=metrics)
            
            mlflow.log_artifact(metrics_dict.filepath, artifact_path="metrics")
            
            for k,v in metrics_dict.data.items():
                if not isinstance(v, list):
                    mlflow.log_metric(k,v)
                else:
                    for step, item in enumerate(v):
                        mlflow.log_metric(k, item, step=step)
        
    def log_plots(self, plots_dir: Union[str, Path]):
        """Log PNG plots from directory.
        
        Args:
            plots_dir: Directory containing PNG files
        """
        plots_dir = Path(plots_dir)
        for plot_file in plots_dir.glob('*.png'):
            mlflow.log_artifact(str(plot_file), artifact_path="plots")
        for plot_file in plots_dir.glob('*.jpg'):
            mlflow.log_artifact(str(plot_file), artifact_path="plots")
        for plot_file in plots_dir.glob('*.html'):
            mlflow.log_artifact(str(plot_file), artifact_path="plots")

    def log_estimator_params(self, est_params: Union[Dict, str, Path]=None):
        """Log estimator parameters from dictionary or JSON file.
        
        Args:
            params: Either dict of parameters or path to JSON file
        """
        if not est_params:
            est_params = self.temp_path/'est_params.json'

        if isinstance(est_params, (str, Path)):
            mlflow.log_artifact(est_params, artifact_path="est_params")
            
        elif isinstance(est_params, dict):
            params_dict = JSON_Handler(filepath=self.temp_path/'est_params.json', data=est_params)

            mlflow.log_artifact(params_dict.filepath, artifact_path="est_params")
            

    def log_dataset(self, dataset: Union[Dict, str, Path]=None):
        """Log dataset reference and optionally the file itself.
        
        Args:
            dataset: Either dict of data or path to JSON file
            log_as_artifact: Whether to upload the dataset file
        """
        if not dataset:
            dataset = self.temp_path/'dataset.json'

        if isinstance(dataset, (str, Path)):
            mlflow.log_artifact(dataset, artifact_path="dataset")
            
        elif isinstance(dataset, dict):
            params_dict = JSON_Handler(filepath=self.temp_path/'dataset.json', data=dataset)

            mlflow.log_artifact(params_dict.filepath, artifact_path="dataset")

    def log_model(self, model, model_format: str = "pickle") -> None:
        """Log trained model with specified format.
        
        Args:
            model: Model object to log
            model_format: Format to use ('pickle', 'keras', etc.)
        """
        if model_format == "pickle":
            mlflow.sklearn.log_model(model, "model")
        elif model_format == "keras":
            mlflow.keras.log_model(model, "model")
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
