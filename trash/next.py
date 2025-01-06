from imports import *

import joblib
import math
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

trainColumns = ['openPrice', 'closePrice', 'highPrice', 'lowPrice', 'volume', 'turnover',
                'startTimeYear', 'startTimeMonth', 'startTimeDay', 'startTimeDayofweek', 'startTimeDayofyear',
                'startTimeIs_month_end', 'startTimeIs_month_start', 'startTimeIs_quarter_end', 'startTimeIs_quarter_start',
                'startTimeIs_year_end', 'startTimeIs_year_start', 'SMA', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff']

def data_process(shift_=15):
    # Загрузка данных
    data = pd.read_csv('BTCUSDT.csv', sep=';')

    # Преобразование временной метки в формат datetime
    data['startTime'] = pd.to_datetime(data['startTime'], unit='ms')

    # Установка индекса по дате
    data.set_index('startTime', inplace=True)

    # Сортировка данных по времени
    data = data.sort_index()

    # Заполнение пропущенных значений методом ffill
    data = data.ffill()

    # Создание нового столбца с ценой закрытия через 15 минут
    if (shift_):
        data[f'PriceClose{shift_}M'] = data['closePrice'].shift(-shift_//15)
        data = data[:-shift_//15]

    def add_datepart(df, fldname, drop=True, time=False, errors="raise"):
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr:
            if n == 'Week':
                df[targ_pre + n] = fld.dt.isocalendar().week
            else:
                df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10**9
        if drop: df.drop(fldname, axis=1, inplace=True)

        df['SMA'] = ta.trend.sma_indicator(df['closePrice'], window=14)
        df['RSI'] = ta.momentum.rsi(df['closePrice'], window=14)
        df['MACD'] = ta.trend.macd(df['closePrice'])
        df['MACD_signal'] = ta.trend.macd_signal(df['closePrice'])
        df['MACD_diff'] = ta.trend.macd_diff(df['closePrice'])

    # Сохранение исходного столбца startTime
    data['startTime'] = data.index
    add_datepart(data, 'startTime')
    data.to_csv(f'BTCUSDT_processed/BTCUSDT_processed{shift_}.csv', sep=';')

    print(data.head())
    print(data.describe())

    # Check for missing columns
    missing_cols = set(trainColumns) - set(data.columns)
    if missing_cols:
        print(f"Warning: Columns {missing_cols} are missing from the data. Filling with 0.")
        for col in missing_cols:
            data[col] = 0

def data_prepare_for_rf_or_gb(shift_=15):
    predictColumn = f'PriceClose{shift_}M'

    data_process(shift_)
    data = pd.read_csv(f'BTCUSDT_processed/BTCUSDT_processed{shift_}.csv', sep=';')

    # Установка индекса по дате
    data.set_index('startTime', inplace=True)
    # Сохранение исходного столбца startTime
    data['startTime'] = data.index

    # Создание новой целевой переменной
    data['Direction'] = (data[predictColumn] > data['closePrice']).astype(int)

    X = data[trainColumns]
    y = data['Direction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, shuffle=False)
    print(X)
    print(y)

    return X_train, X_test, y_train, y_test

def model_(shift_):
    X_train, X_test, y_train, y_test = data_prepare_for_rf_or_gb(shift_)

    # Использование GridSearchCV для оптимизации гиперпараметров
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [20, 30, 40, 50],
        'min_samples_leaf': [ 5, 10, 15],
        'criterion': ['gini', 'entropy']
    }

    clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    # Сохранение лучшей модели
    joblib.dump(best_model, f'models/rf_best_model{shift_}M.pkl')

    # Предсказание направления цен на тестовой выборке
    y_pred = best_model.predict(X_test)

    print(f"Length of predictions: {len(y_pred)}")
    print(f"Length of actual values: {len(y_test)}")

    predictions = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })

    predictions.to_csv('predicted_directions.csv', index=False)

    # Оценка модели
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    accuracy = (predictions['actual'] == predictions['predicted']).sum() / len(predictions)
    print(f"Accuracy of saved predictions: {accuracy:.2f}")

if __name__ == '__main__':
    model_(15)


