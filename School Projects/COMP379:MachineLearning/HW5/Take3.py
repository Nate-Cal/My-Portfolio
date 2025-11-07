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
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_temp)
X_test_scaled = scaler.transform(X_test)

print("Data shapes:")
print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

#Logistic Regression on Test set
print("Logistic Regression on Test Set")

lr_model = LogisticRegression(max_iter=10000, random_state=42)
lr_model.fit(X_train_scaled, y_temp)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Test Accuracy: {lr_acc:.4f}")

# GridSearchCV from scratch
class CustomGridSearchCV:
    def __init__(self, model, param_grid, cv=5):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = 0
        self.cv_results_ = []
    
    def fit(self, X, y):
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Generate all parameter combinations
        param_combinations = list(product(*self.param_grid.values()))
        param_names = list(self.param_grid.keys())
        
        print(f"\nTesting {len(param_combinations)} parameter combinations with {self.cv}-fold CV...")
        
        for idx, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            
            # Skip invalid combinations
            if not self._is_valid_param_combination(params):
                continue
            
            scores = []
            
            # K-Fold Cross Validation
            fold_size = len(X) // self.cv
            
            for i in range(self.cv):
                # Create validation indices
                val_start = i * fold_size
                val_end = (i + 1) * fold_size if i < self.cv - 1 else len(X)
                
                # Split data
                val_indices = np.arange(val_start, val_end)
                train_indices = np.concatenate([np.arange(0, val_start), 
                                               np.arange(val_end, len(X))])
                
                X_train_fold = X[train_indices]
                y_train_fold = y[train_indices]
                X_val_fold = X[val_indices]
                y_val_fold = y[val_indices]
                
                # Train model with current parameters
                try:
                    model_instance = self.model(**params, max_iter=10000, random_state=42)
                    model_instance.fit(X_train_fold, y_train_fold)
                    y_pred = model_instance.predict(X_val_fold)
                    score = accuracy_score(y_val_fold, y_pred)
                    scores.append(score)
                except Exception as e:
                    # Skip this parameter combination if it fails
                    print(f"  Skipping {params}: {str(e)}")
                    break
            
            # Calculate average score across folds
            if len(scores) == self.cv:
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                
                self.cv_results_.append({
                    'params': params,
                    'mean_score': avg_score,
                    'std_score': std_score
                })
                
                print(f"  [{idx+1}/{len(param_combinations)}] {params}")
                print(f"      Score: {avg_score:.4f} (+/- {std_score:.4f})")
                
                # Update best parameters
                if avg_score > self.best_score_:
                    self.best_score_ = avg_score
                    self.best_params_ = params
        
        print(f"\nBest CV Score: {self.best_score_:.4f}")
        print(f"Best Parameters: {self.best_params_}")
    
    def _is_valid_param_combination(self, params):
        """Check if parameter combination is valid for LogisticRegression"""
        # L1 penalty only works with certain solvers
        if params.get('penalty') == 'l1':
            if params.get('solver') not in ['liblinear', 'saga']:
                return False
        
        # lbfgs doesn't support L1
        if params.get('solver') == 'lbfgs':
            if params.get('penalty') == 'l1':
                return False
        
        return True
    
    def get_best_estimator(self):
        """Return the best model trained on full training data"""
        if self.best_params_ is None:
            raise ValueError("Must call fit() before getting best estimator")
        return self.model(**self.best_params_, max_iter=10000, random_state=42)





print("CUSTOM GRID SEARCH WITH CROSS-VALIDATION")


# Define parameter grid 
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],  
    'solver': ['liblinear', 'saga', 'lbfgs']   
}

# Perform custom grid search
custom_grid_search = CustomGridSearchCV(LogisticRegression, param_grid, cv=5)
custom_grid_search.fit(X_train_scaled, y_temp)


# ============================================
# 3. Evaluate Best Model on Test Set
# ============================================
print("\n" + "="*60)
print("3. BEST MODEL EVALUATION")
print("="*60)

# Train best model on full training set
best_model = custom_grid_search.get_best_estimator()
best_model.fit(X_train_scaled, y_temp)

# Test set performance
best_pred_test = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, best_pred_test)

# Training set performance (train and test on training data)
best_pred_train = best_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_temp, best_pred_train)

print(f"Best Parameters: {custom_grid_search.best_params_}")
print(f"\nTraining Set Accuracy: {train_accuracy:.4f}")
print(f"Test Set Accuracy: {test_accuracy:.4f}")
print(f"Difference: {train_accuracy - test_accuracy:.4f}")

print("\n--- Performance Analysis ---")
if train_accuracy > test_accuracy:
    diff = train_accuracy - test_accuracy
    print(f"The model shows signs of overfitting with a {diff:.4f} ({diff*100:.2f}%) difference.")
    print("The model performs better on training data than on unseen test data.")
    if diff > 0.05:
        print("This suggests the model has memorized some training patterns that don't generalize.")
    else:
        print("However, the difference is small, indicating good generalization.")
elif test_accuracy > train_accuracy:
    print("The model generalizes well - test accuracy exceeds training accuracy.")
    print("This is unusual but can happen with small datasets or effective regularization.")
else:
    print("Training and test accuracies are identical - perfect generalization!")

# Display all CV results
print("\n--- All Cross-Validation Results ---")
sorted_results = sorted(custom_grid_search.cv_results_, 
                       key=lambda x: x['mean_score'], reverse=True)
for i, result in enumerate(sorted_results[:5], 1):
    print(f"{i}. {result['params']}")
    print(f"   Score: {result['mean_score']:.4f} (+/- {result['std_score']:.4f})")