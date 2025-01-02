import ta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
import joblib
import time #helper libraries

from data_prepare import *

# Загрузка данных
data_process(0)
input_file = "BTCUSDT_processed/BTCUSDT_processed0.csv"
df = pd.read_csv(input_file, sep=';')


# Расчет технических индикаторов
df['SMA'] = ta.trend.sma_indicator(df['closePrice'], window=14)
df['RSI'] = ta.momentum.rsi(df['closePrice'], window=14)
df['MACD'] = ta.trend.macd(df['closePrice'])
df['MACD_signal'] = ta.trend.macd_signal(df['closePrice'])
df['MACD_diff'] = ta.trend.macd_diff(df['closePrice'])

# Удаление строк с NaN значениями
df.dropna(inplace=True)

# Выбор признаков
features = ['closePrice', 'SMA', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'volume', 'turnover'] #потом можно добавить!!!!!



all_y = df[features].values

# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(all_y)

# Разделение данных на обучающую и тестовую выборки
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Функция для создания набора данных
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])  # Предсказываем только closePrice
    return np.array(dataX), np.array(dataY)

# # Функция для создания набора данных -------------- если через одно
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back - 2):  # Измените здесь на -2
#         a = dataset[i:(i + look_back), :]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back + 1, 0])  # Предсказываем значение через одно
#     return np.array(dataX), np.array(dataY)

look_back = 240
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Изменение формы данных для LSTM
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# Создание и обучение модели
model = Sequential()
model.add(LSTM(5, return_sequences=True, input_shape=(look_back, trainX.shape[2])))  # не 5 а 50 н-р
model.add(Dropout(0.2))
model.add(LSTM(5))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
early_stopping = EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)
model.fit(trainX, trainY, epochs=1, batch_size=240, verbose=1, callbacks=[early_stopping])  #эпох 100-200

# Предсказания
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Обратное преобразование предсказаний
trainPredict = scaler.inverse_transform(np.concatenate((trainPredict, np.zeros((trainPredict.shape[0], trainX.shape[2] - 1))), axis=1))[:, 0]
trainY = scaler.inverse_transform(np.concatenate((trainY.reshape(-1, 1), np.zeros((trainY.shape[0], trainX.shape[2] - 1))), axis=1))[:, 0]
testPredict = scaler.inverse_transform(np.concatenate((testPredict, np.zeros((testPredict.shape[0], testX.shape[2] - 1))), axis=1))[:, 0]
testY = scaler.inverse_transform(np.concatenate((testY.reshape(-1, 1), np.zeros((testY.shape[0], testX.shape[2] - 1))), axis=1))[:, 0]

print(trainY)
print(trainPredict)
print(testY)
print(testPredict)

print(len(testY))
print(len(testPredict))

# Оценка модели
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# Подготовка данных для визуализации
trainPredictPlot = np.full_like(dataset[:, 0], np.nan)
trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict

testPredictPlot = np.full_like(dataset[:, 0], np.nan)
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1] = testPredict

# Визуализация
plt.plot(scaler.inverse_transform(dataset)[:, 0], label='Actual Prices')
plt.plot(trainPredictPlot, label='Train Predictions')
plt.plot(testPredictPlot, label='Test Predictions')
plt.legend()
plt.savefig(f'foolstm.png')
plt.show()

# Сохранение предсказанных данных в файл
testPrices = testY
testPrices = testPrices[:len(testPredict)]  # Убедитесь, что длины совпадают

df_result = pd.DataFrame(data={
    "prediction": np.around(list(testPredict.reshape(-1)), decimals=2),
    "test_price": np.around(list(testPrices.reshape(-1)), decimals=2)
})
df_result.to_csv("lstm_result.csv", sep=';', index=None)
