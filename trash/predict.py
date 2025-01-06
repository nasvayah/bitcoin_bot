from imports import *

# Загрузка модели
best_model = joblib.load('best_model.pkl')

# Загрузка последних 100 значений
data = pd.read_csv('BTCUSDT_last100.csv', sep=';')

# Преобразование временной метки в формат datetime
data['startTime'] = pd.to_datetime(data['startTime'], unit='ms')

# Установка индекса по дате
data.set_index('startTime', inplace=True)

# Сортировка данных по времени
data = data.sort_index()

# Заполнение пропущенных значений методом ffill
data = data.ffill()


# Добавление временных признаков
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

    # Расчет RSI
    def rsi(series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = rsi(df['closePrice'])

    # Расчет MACD
    def macd(series, short_window=12, long_window=26, signal_window=9):
        short_ema = series.ewm(span=short_window, adjust=False).mean()
        long_ema = series.ewm(span=long_window, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        return macd_line, signal_line

    df['MACD_Line'], df['MACD_Signal'] = macd(df['closePrice'])


# Добавление временных признаков и индикаторов
data['startTime'] = data.index
add_datepart(data, 'startTime')

# Definition of trainColumns moved here
trainColumns = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover',
                'startTimeYear', 'startTimeMonth', 'startTimeDay', 'startTimeDayofweek', 'startTimeDayofyear',
                'startTimeIs_month_end', 'startTimeIs_month_start', 'startTimeIs_quarter_end', 'startTimeIs_quarter_start',
                'startTimeIs_year_end', 'startTimeIs_year_start', 'RSI', 'MACD_Line', 'MACD_Signal']

# Check for missing columns
missing_cols = set(trainColumns) - set(data.columns)
if missing_cols:
    print(f"Warning: Columns {missing_cols} are missing from the data. Filling with 0.")
    for col in missing_cols:
        data[col] = 0

print(data.isna().sum() / data.count())

# Пример новых данных для предсказания
X_new = data[trainColumns].tail(1)

# Предсказание новых данных
y_new_pred = best_model.predict(X_new)

# Создание DataFrame с предсказанными ценами
new_predictions = pd.DataFrame({
    '15min': y_new_pred
})

# Сохранение предсказанного значения в CSV файл
new_predictions.to_csv('new_predicted_price.csv', index=False)

print(new_predictions)
