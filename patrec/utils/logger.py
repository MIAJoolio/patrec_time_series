from typing import Literal, Optional, Dict, Any
import logging
import datetime
from pathlib import Path
import json
from contextlib import contextmanager

class Logger:
    """
    Класс для логирования, совместимый с MLflow-подобным интерфейсом.
    """
    def __init__(self, name: str, log_dir: str = ".logs", level: Literal['info', 'debug'] = 'info'):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if level == 'debug' else logging.INFO)
        
        if not self.logger.handlers:
            self._setup_handlers()
            
        self.metrics = {}
        self.params = {}
        
    def _setup_handlers(self):
        """Настройка обработчиков логов."""
        formatter = logging.Formatter(
            "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Файловый обработчик
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_{timestamp}.log"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Логирование метрики."""
        self.metrics[key] = value
        self.logger.info(f"METRIC: {key}={value}" + (f" (step={step})" if step is not None else ""))
        
    def log_param(self, key: str, value: Any):
        """Логирование параметра."""
        self.params[key] = value
        self.logger.info(f"PARAM: {key}={value}")
        
    def log_dict(self, data: Dict[str, Any], prefix: str = ""):
        """Логирование словаря."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self.log_dict(value, full_key)
            else:
                self.log_param(full_key, value)
                
    @contextmanager
    def start_run(self, run_name: Optional[str] = None):
        """Контекстный менеджер для логирования запуска."""
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"run_{run_id}"
        
        self.logger.info(f"Starting run: {run_name}")
        try:
            yield self
        finally:
            self.logger.info(f"Finished run: {run_name}")
            
    def debug(self, msg: str, *args, **kwargs):
        """Логирование отладочного сообщения."""
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs):
        """Логирование информационного сообщения."""
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs):
        """Логирование предупреждения."""
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs):
        """Логирование ошибки."""
        self.logger.error(msg, *args, **kwargs)
        
    def save_metrics(self):
        """Сохранение метрик в JSON файл."""
        metrics_file = self.log_dir / f"{self.name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def save_params(self):
        """Сохранение параметров в JSON файл."""
        params_file = self.log_dir / f"{self.name}_params.json"
        with open(params_file, 'w') as f:
            json.dump(self.params, f, indent=2)

def setup_logger(name: str, log_dir: str = ".logs", level: Literal['info', 'debug'] = 'info') -> Logger:
    """
    Создание и настройка логгера.
    
    Args:
        name: Имя логгера
        log_dir: Директория для сохранения логов
        level: Уровень логирования ('info' или 'debug')
        
    Returns:
        Logger: Настроенный логгер
    """
    return Logger(name, log_dir, level)