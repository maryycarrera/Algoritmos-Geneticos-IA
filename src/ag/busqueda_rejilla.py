from sklearn.model_selection import GridSearchCV
import pandas as pd

from ag.AGEvaluator import AGEvaluator

nombre_dataset_train = 'src/data/toy1_train.csv'
nombre_dataset_val = 'src/data/toy1_val.csv'

datos_train = pd.read_csv(nombre_dataset_train)
X_train = datos_train.iloc[:, :-1].values
y_train = datos_train.iloc[:, -1].values

datos_val = pd.read_csv(nombre_dataset_val)
X_val = datos_val.iloc[:, :-1].values
y_val = datos_val.iloc[:, -1].values

rejilla_hiperparametros = {
    'pc': [0.6, 0.7, 0.8],
    'pm': [0.05, 0.1, 0.2],
    'k': [2, 4, 6],
    'porcentaje_elitismo': [0.1, 0.2, 0.3]
}

busqueda_en_rejilla = GridSearchCV(estimator=AGEvaluator(), param_grid=rejilla_hiperparametros, cv=3, scoring='r2')

busqueda_en_rejilla.fit(X_train, y_train)

best_params = busqueda_en_rejilla.best_params_
best_score = busqueda_en_rejilla.best_score_

print(f'Mejores hiperparámetros: {best_params}')
print(f'Mejor R2 score en entrenamiento: {best_score}')

best_estimator = busqueda_en_rejilla.best_estimator_
validation_score = best_estimator.score(X_val, y_val)
print(f'R2 score en validación: {validation_score}')
