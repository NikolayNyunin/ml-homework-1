import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso

import re
import pickle

def process_torque(torque_str: str) -> (float, float):
    """Функция для обработки столбца `torque`."""

    # если встречен пропуск
    if type(torque_str) is float:
        return np.nan, np.nan

    # удаление запятой, разделяющей разряды некоторых чисел
    torque_str = torque_str.replace(',', '')

    # поиск чисел в строке
    numbers = re.findall(r'\d+\.?\d*', torque_str)

    # первое число - крутящий момент
    torque = float(numbers[0])
    # конвертация единиц измерения при необходимости
    if 'kgm' in torque_str.lower() and 'nm' not in torque_str.lower():
        # если указано только значение в `kgm`, то конвертируем его
        torque *= 9.81

    # если нет данных об оборотах
    if len(numbers) == 1:
        return torque, np.nan

    # иногда второе значение - крутящий момент в `kgm`, удаляем его
    if '.' in numbers[1]:
        numbers.pop(1)

    # второе число (или 2-е и 3-е) - обороты двигателя, при которых достигается крутящий момент
    rpm = sum(map(float, numbers[1:])) / len(numbers[1:])

    return torque, rpm


class CustomNumTransformer(BaseEstimator, TransformerMixin):
    """Кастомный класс для преобразования числовых признаков."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['mileage'] = X['mileage'].str.split().str[0].astype(float)
        X['engine'] = X['engine'].str.split().str[0].astype(float)
        X['max_power'] = X['max_power'].str.replace('bhp', 'NaN')
        X['max_power'] = X['max_power'].str.split().str[0].astype(float)
        X['torque'] = X['torque'].apply(process_torque)
        X['max_torque_rpm'] = X['torque'].str[1].astype(float)
        X['torque'] = X['torque'].str[0].astype(float)
        return X


class CustomCatTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['name'] = X['name'].str.split().str[0]
        return X


def fit_and_save_pipeline(path='pipeline.pickle') -> None:
    """Получение обученного пайплайна."""

    # чтение датасета
    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

    # удаление дубликатов
    df_train = df_train[df_train.drop('selling_price', axis=1).duplicated() == False]
    df_train = df_train.reset_index(drop=True)

    # проверка размерности
    assert df_train.shape == (5840, 13)

    X_train, y_train = df_train.drop('selling_price', axis=1), df_train['selling_price']
    X_test, y_test = df_test.drop('selling_price', axis=1), df_test['selling_price']

    num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque']
    num_preprocessor = Pipeline(steps=[
        ('custom_transformer', CustomNumTransformer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
        ('poly', PolynomialFeatures(degree=2))
    ])

    cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
    cat_preprocessor = Pipeline(steps=[
        ('custom_transformer', CustomCatTransformer()),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    col_transformer = ColumnTransformer([
        ('num_preprocessor', num_preprocessor, num_cols),
        ('cat_preprocessor', cat_preprocessor, cat_cols)
    ])

    lasso_pipeline = Pipeline(steps=[
        ('preprocessor', col_transformer),
        ('classifier', Lasso())
    ])

    lasso_pipeline.fit(X_train, y_train)
    print(f'Пайплайн обучен, R2 = {lasso_pipeline.score(X_test, y_test):.5f}')

    with open(path, 'wb') as file:
        pickle.dump(lasso_pipeline, file)
        print(f'Пайплайн сохранён как "{path}"')


if __name__ == '__main__':
    fit_and_save_pipeline()
