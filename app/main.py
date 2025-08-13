from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import logging
from .schemas import JSON_Handler, TimeSeriesData, DatasetInfo
from .utils import load_ucr_dataset, load_uea_dataset, validate_time_series
from .logger import setup_logger

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logger = setup_logger()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/api/datasets/upload/")
async def upload_datasets(files: List[UploadFile] = File(...)):
    """
    Загрузка датасетов временных рядов
    
    Параметры:
    - files: Список файлов в формате CSV, JSON или NPY
    
    Возвращает:
    - JSON с информацией о загруженных датасетах
    """
    results = {}
    json_handler = JSON_Handler(datasets={}, metadata={})
    
    for file in files:
        try:
            content = await file.read()
            dataset_name = file.filename.split('.')[0]
            
            # Валидация и загрузка данных
            if file.filename.endswith('.npy'):
                data = np.load(BytesIO(content))
                ts_data = validate_time_series(data)
            elif file.filename.endswith('.csv'):
                df = pd.read_csv(BytesIO(content))
                ts_data = validate_time_series(df.values)
            elif file.filename.endswith('.json'):
                ts_data = TimeSeriesData.parse_raw(content)
            else:
                raise HTTPException(400, "Unsupported file format")
            
            # Добавляем в JSON_Handler
            json_handler.datasets[dataset_name] = ts_data
            json_handler.metadata[dataset_name] = DatasetInfo(
                name=dataset_name,
                source="uploaded",
                n_samples=len(ts_data.values),
                n_features=1,  # Для одномерных рядов
                length=len(ts_data.values),
                n_classes=len(set(ts_data.labels)) if ts_data.labels else None
            )
            
            results[dataset_name] = {"status": "success", "samples": len(ts_data.values)}
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results[file.filename] = {"status": "error", "message": str(e)}
    
    return JSONResponse(content={"datasets": results, "handler": json_handler.dict()})

@app.get("/api/datasets/ucr/{dataset_name}")
async def get_ucr_dataset(dataset_name: str):
    """
    Загрузка датасета из UCR репозитория
    
    Параметры:
    - dataset_name: Имя датасета из UCR
    
    Возвращает:
    - JSON с данными временных рядов
    """
    try:
        data, labels = load_ucr_dataset(dataset_name)
        ts_data = TimeSeriesData(name=dataset_name, values=data.tolist(), labels=labels.tolist())
        
        json_handler = JSON_Handler(
            datasets={dataset_name: ts_data},
            metadata={
                dataset_name: DatasetInfo(
                    name=dataset_name,
                    source="UCR",
                    n_samples=len(data),
                    n_features=1,
                    length=len(data[0]),
                    n_classes=len(set(labels))
            }
        )
        
        return JSONResponse(content=json_handler.dict())
    except Exception as e:
        logger.error(f"Error loading UCR dataset {dataset_name}: {str(e)}")
        raise HTTPException(404, f"Dataset {dataset_name} not found or error loading: {str(e)}")

@app.get("/api/datasets/uea/{dataset_name}")
async def get_uea_dataset(dataset_name: str):
    """
    Загрузка датасета из UEA репозитория
    
    Параметры:
    - dataset_name: Имя датасета из UEA
    
    Возвращает:
    - JSON с данными временных рядов
    """
    try:
        data, labels = load_uea_dataset(dataset_name)
        ts_data = TimeSeriesData(name=dataset_name, values=data.tolist(), labels=labels.tolist())
        
        json_handler = JSON_Handler(
            datasets={dataset_name: ts_data},
            metadata={
                dataset_name: DatasetInfo(
                    name=dataset_name,
                    source="UEA",
                    n_samples=len(data),
                    n_features=data.shape[1],
                    length=data.shape[2]),
                    n_classes=len(set(labels))
            }
        )
        
        return JSONResponse(content=json_handler.dict())
    except Exception as e:
        logger.error(f"Error loading UEA dataset {dataset_name}: {str(e)}")
        raise HTTPException(404, f"Dataset {dataset_name} not found or error loading: {str(e)}")