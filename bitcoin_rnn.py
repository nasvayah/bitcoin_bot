from imports import * # Assuming this line imports necessary libraries (pandas, numpy, tensorflow, sklearn, etc.)
from data_prepare import * # Assumes this imports your data preparation functions (including RSI, MACD calculation).




def model_(look_back=24, shift_=15):
    X_train, X_test, y_train, y_test, scalerX, scalery, test_indices, data  = prepare_data_for_lstm(look_back, shift_)

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)  # Increase epochs if needed

    # Predict
    y_pred = model.predict(X_test)
    print(y_pred)


    # Inverse transform to get percentage change values
    y_test_percentage = scalery.inverse_transform(y_test)
    y_pred_percentage = scalery.inverse_transform(y_pred)

    # Ensure y_test_percentage and y_pred_percentage are 1-dimensional
    y_test_percentage = y_test_percentage.flatten()
    y_pred_percentage = y_pred_percentage.flatten()


        # Calculate the actual prices from the percentage change
    y_test_original = []
    y_pred_original = []
    for i in range(len(y_test_percentage)):
        actual_price = data.iloc[look_back + len(X_train) + i]['closePrice']
        y_test_original.append(actual_price * (1 + y_test_percentage[i] / 100))
        y_pred_original.append(actual_price * (1 + y_pred_percentage[i] / 100))


    predictions = pd.DataFrame({
        'actual': y_test_original,
        'predicted': y_pred_original
    }, index=test_indices)

    predictions.to_csv('predicted_prices.csv', index=False)
    joblib.dump(model, f'models/rnn_best_model{shift_}M.pkl')

    plot_data = predictions[:100]
    plot_data = plot_data[10:]
    plot_data.plot()
    plt.savefig(f'foos/foo{shift_}M.png')
    plt.show()

if __name__ == '__main__':
    model_(24, 15)
