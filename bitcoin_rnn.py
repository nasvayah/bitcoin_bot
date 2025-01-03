from imports import * # Assuming this line imports necessary libraries (pandas, numpy, tensorflow, sklearn, etc.)
from data_prepare import * # Assumes this imports your data preparation functions (including RSI, MACD calculation).





def model_(look_back=24, shift_=15):
    X_train, X_test, y_train, y_test, scalerX, scalery, test_indices  = prepare_data_for_lstm(look_back, shift_)

    # Create LSTM model
    model = Sequential()
    

    model.add(LSTM(units=200, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(1))


    model.compile(optimizer='adam', loss='mean_squared_error')
    

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

    # Predict
    y_pred = model.predict(X_test)



    # Inverse transform to get actual price values
    y_test_original = scalery.inverse_transform(y_test)
    y_pred_original = scalery.inverse_transform(y_pred)

        # Ensure y_test_original and y_pred_original are 1-dimensional
    y_test_original = y_test_original.flatten()
    y_pred_original = y_pred_original.flatten()


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


