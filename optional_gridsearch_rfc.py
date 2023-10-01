from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
dataset = pandas.read_csv('breast_cancer_.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)


estimator = RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(10, 160, 10)]
criterion = ['gini', 'entropy', 'log_loss']
min_samples_split = [2, 4, 8, 16]
bootstrap = [True, False]
random_state = [int(x) for x in np.linspace(0, 21)]
max_samples = [0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8]
max_leaf_nodes = [2, 4, 8, 16, 32, 64, 128, None]
param_grid = {
    'n_estimators': n_estimators,
    'criterion': criterion,
    'min_samples_split': min_samples_split,
    'bootstrap': bootstrap,
    'random_state': random_state,
    'max_leaf_nodes': max_leaf_nodes,
    'max_samples': max_samples
}

grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, scoring='accuracy', cv=3)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_, grid_search.best_score_)
