from models import fit_and_save_pipeline

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from typing import List
import os
import pickle
from contextlib import asynccontextmanager

PIPELINE_PATH = 'pipeline.pickle'


class Info(BaseModel):
    name: str
    description: str


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    items: List[Item]


class Prediction(BaseModel):
    selling_price: float


class Predictions(BaseModel):
    predictions: List[Prediction]


def get_predictions(data: pd.DataFrame) -> np.ndarray:
    """Получение предсказания модели."""

    try:
        return np.round(ml_models['pipeline'].predict(data), 2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e).capitalize())


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекстный менеджер для ML-модели приложения."""

    # если файла с пайплайном нет, вызываем функцию для его создания
    if not os.path.exists(PIPELINE_PATH):
        fit_and_save_pipeline(PIPELINE_PATH)

    # загружаем пайплайн из Pickle-файла
    with open(PIPELINE_PATH, 'rb') as file:
        pipeline = pickle.load(file)

    # сохраняем полученный пайплайн
    ml_models['pipeline'] = pipeline

    yield

    # удаляем пайплайн из памяти (ну почти)
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.get('/', response_model=Info)
async def root():
    """Общая информация о приложении."""

    return {'name': 'Car Prices Prediction',
            'description': 'FastAPI-based ML service for predicting car prices by their features'}


@app.post('/predict_item', response_model=Prediction)
async def predict_item(item: Item):
    """Получить предсказание для одного объекта."""

    data = pd.DataFrame([jsonable_encoder(item)])
    return {'selling_price': get_predictions(data)[0]}


@app.post('/predict_items', response_model=Predictions)
async def predict_items(items: Items):
    """Получить предсказания для нескольких объектов."""

    data = pd.DataFrame(jsonable_encoder(items)['items'])
    return {'predictions': [{'selling_price': pred} for pred in get_predictions(data)]}
