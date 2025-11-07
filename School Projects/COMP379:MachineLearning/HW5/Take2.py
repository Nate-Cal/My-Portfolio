import pandas as pd
from itertools import product
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 


# URL
url = 'wine/wine.data'
data = pd.read_csv(url)

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split dataset: 80% train, 20% test
# 80% train, 20% test
# Split off training set 
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_temp)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression on test set
lr_model = LogisticRegression(max_iter=10000, random_state=42)
lr_model.fit(X_train_scaled, y_temp)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)

print("\n--- Logistic Regression on Test Set ---")
print(f"Accuracy: {lr_acc:.3f}") 

# GridSearchCV from scratch
class CustomGridSearchCV:
    def __init__(self, model, param_grid, cv=5):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = 0

    def fit(self, X, y):
        

        param_combinations = list(product(*self.param_grid.values()))
        param_names = list(self.param_grid.keys())

        for param_values in param_combinations:
            params = dict(zip(param_names, param_values))
            scores = []

            # K-Fold Cross Validation
            fold_size = len(X) // self.cv
            for i in range(self.cv):
                X_val = X[i*fold_size:(i+1)*fold_size]
                y_val = y[i*fold_size:(i+1)*fold_size]
                X_train = pd.concat([X[:i*fold_size], X[(i+1)*fold_size:]])
                y_train = pd.concat([y[:i*fold_size], y[(i+1)*fold_size:]])

                model_instance = self.model(**params)
                model_instance.fit(X_train, y_train)
                y_pred = model_instance.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

            avg_score = np.mean(scores)
            if avg_score > self.best_score_:
                self.best_score_ = avg_score
                self.best_params_ = params

    def best_estimator_(self):
        return self.model(**self.best_params_)
    
# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga', 'lbfgs']
}   

# Perform custom grid search
custom_grid_search = CustomGridSearchCV(LogisticRegression, param_grid, cv=5)
custom_grid_search.fit(pd.DataFrame(X_train_scaled), pd.Series(y_temp))
best_model = custom_grid_search.best_estimator_()
best_model.fit(X_train_scaled, y_temp)
best_pred = best_model.predict(X_test_scaled)
best_acc = accuracy_score(y_test, best_pred)

print("\n--- Logistic Regression with Custom GridSearchCV ---")
print(f"Best Parameters: {custom_grid_search.best_params_}")
print(f"Accuracy: {best_acc:.3f}")
