import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# TODO: CHANGE URL TO LOCATION OF WINE.DATA FILE
# Load the dataset
url = 'wine/wine.data'
data = pd.read_csv(url)

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# 80% train, 20% test
# Split off training set 
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_temp)} ({len(y_temp)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Standardize features
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

# Improving LR with GridSearchCV
# Need to implement GridSearchCV from scratch and compare results to your 
# manual hyperparameter tuning above.
# You are required to implement your own
# n-fold cross-validation from scratch rather than using the scikit-learn
# API. I.e. you will need to randomly split the training set into n parts
# and repeatedly train on n-1 parts and test on the remaining part

best_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga', 'lbfgs']
}
grid_search = GridSearchCV(LogisticRegression(max_iter=10000, random_state=42), 
                           best_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_temp)
grid_best = grid_search.best_estimator_
grid_best_pred = grid_best.predict(X_test_scaled)
grid_best_acc = accuracy_score(y_test, grid_best_pred)

print("\n--- Logistic Regression with GridSearchCV ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {grid_best_acc:.3f}")