from imports import *
from data_prepare import *

def model_(shift_=15):
    X_train, X_test, y_train, y_test = data_prepare_for_rf_or_gb(shift_)
        # Заполнение пропущенных значений методом ffill
    X_train = X_train.ffill()
    X_test = X_test.ffill()

    # Использование импутера для заполнения оставшихся пропущенных значений
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    train_error = []
    test_error = []
    minDepth = 40
    maxDepth = 45
    models = []
    for depth in range(minDepth, maxDepth, 5):
        regr = GradientBoostingRegressor(max_depth=3, random_state=0, n_estimators=100, verbose=2,learning_rate=0.05)
        regr.fit(X_train, y_train)
        models.append(regr)
        tr_error = math.sqrt(mean_squared_error(regr.predict(X_train), y_train))
        te_error = math.sqrt(mean_squared_error(regr.predict(X_test), y_test))
        test_error.append(tr_error)
        train_error.append(te_error)
        print(depth, tr_error, te_error)

    # Предсказание цен на тестовой выборке
    best_model = models[-1]  # Используем последнюю модель с максимальной глубиной
    y_pred = best_model.predict(X_test)
    joblib.dump(best_model, f'models/gb_best_model{shift_}M.pkl')

    predictions = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })

    # Сохранение предсказанных цен в CSV файл
    #predictions.to_csv('predicted_prices.csv', index=False)

    plot_data = predictions[:100]
    plot_data = plot_data[10:]
    plot_data.plot()
    plt.savefig(f'foos/foo{shift_}M.png')
    plt.show()


if __name__ == '__main__':
    model_(30)