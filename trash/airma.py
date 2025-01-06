import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import joblib
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib


trainColumns = ['openPrice', 'closePrice', 'highPrice', 'lowPrice', 'volume', 'turnover',
                    'Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear',
                    'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start',
                    'Is_year_end', 'Is_year_start', 'SMA', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff']

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
    if shift_:
        data[f'PriceClose{shift_}M'] = data['closePrice'].shift(-shift_ // 15)
        data = data[:-shift_ // 15]

    # Сохранение исходного столбца startTime
    data['startTime'] = data.index

    # Добавление временных признаков
    data['Year'] = data['startTime'].dt.year
    data['Month'] = data['startTime'].dt.month
    data['Day'] = data['startTime'].dt.day
    data['Dayofweek'] = data['startTime'].dt.dayofweek
    data['Dayofyear'] = data['startTime'].dt.dayofyear
    data['Is_month_end'] = data['startTime'].dt.is_month_end
    data['Is_month_start'] = data['startTime'].dt.is_month_start
    data['Is_quarter_end'] = data['startTime'].dt.is_quarter_end
    data['Is_quarter_start'] = data['startTime'].dt.is_quarter_start
    data['Is_year_end'] = data['startTime'].dt.is_year_end
    data['Is_year_start'] = data['startTime'].dt.is_year_start

    # Добавление финансовых индикаторов
    data['SMA'] = ta.trend.sma_indicator(data['closePrice'], window=14)
    data['RSI'] = ta.momentum.rsi(data['closePrice'], window=14)
    data['MACD'] = ta.trend.macd(data['closePrice'])
    data['MACD_signal'] = ta.trend.macd_signal(data['closePrice'])
    data['MACD_diff'] = ta.trend.macd_diff(data['closePrice'])

    data.to_csv(f'BTCUSDT_processed/BTCUSDT_processed{shift_}.csv', sep=';')

    print(data.head())
    print(data.describe())

    # Check for missing columns
    missing_cols = set(trainColumns) - set(data.columns)
    if missing_cols:
        print(f"Warning: Columns {missing_cols} are missing from the data. Filling with 0.")
        for col in missing_cols:
            data[col] = 0

def data_prepare_for_arima(shift_=15):
    data_process(shift_)
    data = pd.read_csv(f'BTCUSDT_processed/BTCUSDT_processed{shift_}.csv', sep=';')

    # Установка индекса по дате
    data.set_index('startTime', inplace=True)

    # Сохранение исходного столбца startTime
    data['startTime'] = data.index

    # Использование только closePrice для ARIMA
    y = data[f'PriceClose{shift_}M']

    # Разделение данных на тренировочную и тестовую выборки
    train_size = int(len(y) * 0.999)
    train, test = y[:train_size], y[train_size:]

    return train, test

def model_arima(shift_):
    train, test = data_prepare_for_arima(shift_)

    # Преобразование данных для анализа тренда
    train_diff = train.diff().dropna()
    test_diff = test.diff().dropna()



    # Обучение модели ARIMA
    model = ARIMA(train_diff, order=(5, 1, 0))  # Пример параметров (p, d, q)
    model_fit = model.fit()

    # Сохранение модели
    joblib.dump(model_fit, f'models/arima_best_model{shift_}M.pkl')

    # Предсказание цен на тестовой выборке
    y_pred = []
    history = train_diff.tolist()

    for t in range(len(test_diff)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        y_pred.append(output[0])
        history.append(test_diff[t])

    print(f"Length of predictions: {len(y_pred)}")
    print(f"Length of actual values: {len(test_diff)}")

    predictions = pd.DataFrame({
        'actual': test_diff,
        'predicted': y_pred
    })

    predictions.to_csv('predicted_prices.csv', index=False)

    # Визуализация результатов
    plot_data = predictions[:100]
    plot_data = plot_data[80:]
    plot_data.plot()
    plt.title(f'Predicted vs Actual Prices (Shift: {shift_}M)')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend(['Actual', 'Predicted'])
    plt.savefig(f'foos/foo{shift_}M.png')
    plt.show()

if __name__ == '__main__':
    model_arima(15)