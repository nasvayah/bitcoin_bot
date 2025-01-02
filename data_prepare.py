from imports import *

trainColumns = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover',
                    'startTimeYear', 'startTimeMonth', 'startTimeDay', 'startTimeDayofweek', 'startTimeDayofyear',
                    'startTimeIs_month_end', 'startTimeIs_month_start', 'startTimeIs_quarter_end', 'startTimeIs_quarter_start',
                    'startTimeIs_year_end', 'startTimeIs_year_start', 'RSI', 'MACD_Line', 'MACD_Signal']

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


    # Сохранение исходного столбца startTime
    data['startTime'] = data.index
    add_datepart(data, 'startTime')
    data.to_csv(f'BTCUSDT_processed/BTCUSDT_processed{shift_}.csv', sep=';')

    print(data.head())
    print(data.describe())

    # # Definition of trainColumns moved here
    # trainColumns = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover',
    #                 'startTimeYear', 'startTimeMonth', 'startTimeDay', 'startTimeDayofweek', 'startTimeDayofyear',
    #                 'startTimeIs_month_end', 'startTimeIs_month_start', 'startTimeIs_quarter_end', 'startTimeIs_quarter_start',
    #                 'startTimeIs_year_end', 'startTimeIs_year_start', 'RSI', 'MACD_Line', 'MACD_Signal']
    # predictColumn = f'PriceClose{shift_}M'

    # Check for missing columns
    missing_cols = set(trainColumns) - set(data.columns)
    if missing_cols:
        print(f"Warning: Columns {missing_cols} are missing from the data. Filling with 0.")
        for col in missing_cols:
            data[col] = 0
    # print(data.isna().sum() / data.count())



# Загрузка данных
def data_prepare_for_rf_or_gb(shift_=15):

    predictColumn = f'PriceClose{shift_}M'


    data_process(shift_)
    data = pd.read_csv(f'BTCUSDT_processed/BTCUSDT_processed{shift_}.csv',sep=';')

 
        # Установка индекс по дате
    data.set_index('startTime', inplace=True)
    # Сохранение исходного столбца startTime
    data['startTime'] = data.index

    X = data[trainColumns]
    y = data[predictColumn]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, shuffle=False)
    print(X)
    print(y)

    return X_train, X_test, y_train, y_test 






def prepare_data_for_lstm(look_back=24, shift_=15): #look_back is the number of past hours to use as input for each prediction.
    # Load data
    data_process(shift_)
    data = pd.read_csv(f'BTCUSDT_processed/BTCUSDT_processed{shift_}.csv', sep=';')
    data['startTime'] = pd.to_datetime(data['startTime'])
    data = data.set_index('startTime')

    # Feature Engineering (assuming you have calculated RSI and MACD in data_prepare)
    trainColumns = ['closePrice', 'volume', 'turnover',
                    'startTimeYear', 'startTimeMonth', 'startTimeDay', 'startTimeDayofweek', 'startTimeDayofyear',
                    'startTimeIs_month_end', 'startTimeIs_month_start', 'startTimeIs_quarter_end', 'startTimeIs_quarter_start',
                    'startTimeIs_year_end', 'startTimeIs_year_start', 'RSI', 'MACD_Line', 'MACD_Signal']
    predictColumn = f'PriceClose{shift_}M'

    # Handle missing data (check if any exist after running data_prepare)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)  # Additional step to ensure no NaN values remain
        # Check for NaN values in the original data
    if data.isnull().values.any():
        raise ValueError("NaN values found in the original data")

    # Normalize data
    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalery = MinMaxScaler(feature_range=(0, 1))
    data_X_to_scale = data[trainColumns]
    data_y_to_scale = data[[predictColumn]]  # Convert to DataFrame
    scaled_data_X = scalerX.fit_transform(data_X_to_scale)
    scaled_data_y = scalery.fit_transform(data_y_to_scale)


        # Check for NaN values after scaling
    if np.isnan(scaled_data_X).any() or np.isnan(scaled_data_y).any():
        raise ValueError("NaN values found after scaling")

    # Prepare data for LSTM
    X, y = [], []
    for i in range(look_back, len(scaled_data_X)):
        X.append(scaled_data_X[i - look_back:i])
        y.append(scaled_data_y[i])
    X, y = np.array(X), np.array(y)


        # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, shuffle=False)


        # Save the test indices for later use
    test_indices = data.index[look_back + len(X_train):look_back + len(X_train) + len(X_test)]

    return X_train, X_test, y_train, y_test, scalerX, scalery, test_indices