from imports import *
from data_prepare import *


def model_(shift_):
    X_train, X_test, y_train, y_test = data_prepare_for_rf_or_gb(shift_)
    
    train_error = []
    test_error = []
    minDepth = 40
    maxDepth = 45
    models = []
    for depth in range(minDepth,maxDepth , 5):
        regr = RandomForestRegressor(max_depth=depth, random_state=0, n_estimators=50, verbose=2, n_jobs=-1, min_samples_split=5, min_samples_leaf=4)
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
    joblib.dump(best_model, f'models/rf_best_model{shift_}M.pkl')


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


































# from sklearn.model_selection import GridSearchCV

# # Определение параметров для GridSearchCV
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [20, 30, 40],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }

# # Создание модели RandomForestRegressor
# rf = RandomForestRegressor(random_state=0, n_jobs=-1)

# # Использование GridSearchCV для настройки гиперпараметров
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# # Лучшие параметры
# best_params =  grid_search.best_params_
# print("Best Parameters:", best_params)

# # Обучение модели с лучшими параметрами
# best_rf = grid_search.best_estimator_
# best_rf.fit(X_train, y_train)

# train_plot = pd.DataFrame(train_error, index=range(20, 40, 5), columns=["test_Data"])
# test_plot = pd.DataFrame(test_error, index=range(20, 40, 5), columns=["train_Data"])
# plotdata = pd.concat([train_plot, test_plot], axis=1)
# plotdata.plot()
# plt.show()