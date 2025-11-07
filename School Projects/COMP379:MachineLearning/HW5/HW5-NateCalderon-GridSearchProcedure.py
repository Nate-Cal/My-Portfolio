import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
url = 'wine/wine.data'
data = pd.read_csv(url)

# Separate features and labels
X = data.drop('Class', axis=1).values
Y = data['Class'].values

# Split Data -> 80% Train, 20% Test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# N-Fold Cross-Validation
class NFoldCrossValidation:
    def __init__(self, n_folds=5):
        # Initialize n-fold CV
        self.n_folds = n_folds

    def randomize_folds(self, X, Y):
        # Create randomized folds
        n_samples = len(X)
        # Initialize shuffled indices and randomize
        shuffled_indices = np.random.permutation(n_samples)
        # Initialize fold size and folds list using n_folds
        fold_size = n_samples // self.n_folds
        # Create empty folds list
        folds = []
        # Create folds
        for i in range(self.n_folds):
            # Determine fold indices
            start = i * fold_size
            # Ensure last fold takes remaining samples
            end = (i + 1) * fold_size if i < self.n_folds - 1 else n_samples
            #  Initialize fold indices
            fold_indices = shuffled_indices[start:end]
            # Select fold data
            folds.append((X[fold_indices], Y[fold_indices]))
        return folds

    def split(self, folds, val_index):
        # Split folds into training and validation sets
        X_val, Y_val = folds[val_index]
        X_train = np.vstack([folds[i][0] for i in range(self.n_folds) if i != val_index])
        Y_train = np.concatenate([folds[i][1] for i in range(self.n_folds) if i != val_index])
        return X_train, Y_train, X_val, Y_val

    def train(self, X, Y, model_class, **model_params):
        folds = self.randomize_folds(X, Y)
        fold_accuracies = []

        for i in range(self.n_folds):
            # Split: one fold for validation, others for training
            X_train, Y_train, X_val, Y_val = self.split(folds, i)

            # Train model
            model = model_class(**model_params)
            model.fit(X_train, Y_train)

            # Validate model
            Y_pred = model.predict(X_val)
            acc = accuracy_score(Y_val, Y_pred)
            fold_accuracies.append(acc)

            print(f"Fold {i+1}/{self.n_folds} accuracy: {acc:.3f}")

        avg_acc = np.mean(fold_accuracies)
        print(f"\nAverage cross-validation accuracy: {avg_acc:.3f}")
        return avg_acc

# Custom Grid Search Implementation
class CustomGridSearch:
    def __init__(self, model_class, param_grid, cv_class):
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv_class = cv_class

    def fit(self, X, Y):
        best_score = -1
        best_params = None
        results = []

        for C_val in self.param_grid['C']:
            for penalty_val in self.param_grid['penalty']:
                
                print(f"Testing C={C_val}, penalty='{penalty_val}'")

                avg_acc = self.cv_class.train(
                    X, Y,
                    self.model_class,
                    C=C_val,
                    penalty=penalty_val,
                    max_iter=10000
                )

                results.append({
                    'C': C_val,
                    'penalty': penalty_val,
                    'avg_accuracy': avg_acc
                })

                if avg_acc > best_score:
                    best_score = avg_acc
                    best_params = {'C': C_val, 'penalty': penalty_val}

        print(f"Best parameters: {best_params}")
        print(f"Best average CV accuracy: {best_score:.3f}\n")

        self.best_params = best_params
        self.best_score = best_score
        self.results = pd.DataFrame(results)
        return self

# Run custom grid search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2']
}

cv = NFoldCrossValidation(n_folds=5)
grid_search = CustomGridSearch(LogisticRegression, param_grid, cv)
grid_search.fit(X_train_scaled, Y_train)

# Train final model with best parameters
best_params = grid_search.best_params
final_model = LogisticRegression(
    C=best_params['C'],
    penalty=best_params['penalty'],
    max_iter=10000
)
final_model.fit(X_train_scaled, Y_train)

# Evaluate on test set
Y_pred_test = final_model.predict(X_test_scaled)
test_acc = accuracy_score(Y_test, Y_pred_test)
print(f"Test accuracy with best parameters: {test_acc:.3f}")
