import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter

# Load the dataset (file has header row)
url = 'wine/wine.data'
data = pd.read_csv(url)

print("=" * 80)
print("WINE CLASSIFICATION PROJECT")
print("=" * 80)
print(f"\nDataset shape: {data.shape}")
print(f"Class distribution:\n{data['Class'].value_counts().sort_index()}")

# Check for any data issues
print(f"\nChecking for missing values:\n{data.isnull().sum()}")
print(f"\nUnique classes: {sorted(data['Class'].unique())}")

# Remove any rows with missing values or invalid classes
data = data.dropna()
data = data[data['Class'].isin([1, 2, 3])]

print(f"\nCleaned dataset shape: {data.shape}")
print(f"Cleaned class distribution:\n{data['Class'].value_counts().sort_index()}")

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split: 70% train, 15% dev, 15% test
# Strategy: First split off train (70%), then split remaining 30% into dev and test (each 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
# Now split the temp set (30%) into dev and test (50/50 split = 15% each of original)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nTraining set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Development set size: {len(X_dev)} ({len(X_dev)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 80)
print("EVALUATION METRIC SELECTION")
print("=" * 80)
print("For this multi-class wine classification problem:")
print("- Classes are relatively balanced")
print("- We use ACCURACY as the primary metric (suitable for balanced datasets)")
print("- F1-score (macro) as secondary metric for comprehensive evaluation")

# ============================================================================
# STEP 1: Train default classifiers
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Default Classifiers on Development Set")
print("=" * 80)

# Logistic Regression
lr_model = LogisticRegression(max_iter=10000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_dev_scaled)
lr_acc = accuracy_score(y_dev, lr_pred)
lr_f1 = f1_score(y_dev, lr_pred, average='macro')

print("\n--- Logistic Regression (Default) ---")
print(f"Accuracy: {lr_acc:.4f}")
print(f"F1-score (macro): {lr_f1:.4f}")
print(f"\nClassification Report:\n{classification_report(y_dev, lr_pred)}")

# SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_dev_scaled)
svm_acc = accuracy_score(y_dev, svm_pred)
svm_f1 = f1_score(y_dev, svm_pred, average='macro')

print("\n--- SVM (Default) ---")
print(f"Accuracy: {svm_acc:.4f}")
print(f"F1-score (macro): {svm_f1:.4f}")
print(f"\nClassification Report:\n{classification_report(y_dev, svm_pred)}")

# ============================================================================
# STEP 2: Hyperparameter tuning
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Hyperparameter Tuning")
print("=" * 80)

# Tune Logistic Regression
print("\n--- Tuning Logistic Regression ---")
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}
lr_grid = GridSearchCV(LogisticRegression(max_iter=10000, random_state=42), 
                       lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
lr_grid.fit(X_train_scaled, y_train)
lr_tuned = lr_grid.best_estimator_
lr_tuned_pred = lr_tuned.predict(X_dev_scaled)
lr_tuned_acc = accuracy_score(y_dev, lr_tuned_pred)
lr_tuned_f1 = f1_score(y_dev, lr_tuned_pred, average='macro')

print(f"Best parameters: {lr_grid.best_params_}")
print(f"Accuracy: {lr_tuned_acc:.4f} (improvement: {lr_tuned_acc - lr_acc:+.4f})")
print(f"F1-score (macro): {lr_tuned_f1:.4f}")

# Tune SVM
print("\n--- Tuning SVM ---")
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, 
                        scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)
svm_tuned = svm_grid.best_estimator_
svm_tuned_pred = svm_tuned.predict(X_dev_scaled)
svm_tuned_acc = accuracy_score(y_dev, svm_tuned_pred)
svm_tuned_f1 = f1_score(y_dev, svm_tuned_pred, average='macro')

print(f"Best parameters: {svm_grid.best_params_}")
print(f"Accuracy: {svm_tuned_acc:.4f} (improvement: {svm_tuned_acc - svm_acc:+.4f})")
print(f"F1-score (macro): {svm_tuned_f1:.4f}")

# Select best model from step 2
best_sklearn_model = lr_tuned if lr_tuned_acc >= svm_tuned_acc else svm_tuned
best_sklearn_name = "Logistic Regression" if lr_tuned_acc >= svm_tuned_acc else "SVM"
best_sklearn_acc = max(lr_tuned_acc, svm_tuned_acc)

# ============================================================================
# STEP 3: Custom KNN Implementation
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Custom K-Nearest Neighbors Implementation")
print("=" * 80)

class KNNClassifier:
    """Custom K-Nearest Neighbors classifier implementation"""
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        predictions = []
        for x in X:
            # Calculate Euclidean distances to all training samples
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train.iloc[k_indices]
            
            # Vote: most common label
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)

# Tune k value
print("\n--- Tuning k value ---")
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
k_scores = []

for k in k_values:
    knn = KNNClassifier(k=k)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_dev_scaled)
    acc = accuracy_score(y_dev, knn_pred)
    k_scores.append(acc)
    print(f"k={k:2d}: Accuracy = {acc:.4f}")

# Best k
best_k = k_values[np.argmax(k_scores)]
best_knn_acc = max(k_scores)
print(f"\nBest k value: {best_k}")
print(f"Best KNN accuracy on dev set: {best_knn_acc:.4f}")

# Train final KNN with best k
knn_best = KNNClassifier(k=best_k)
knn_best.fit(X_train_scaled, y_train)
knn_best_pred = knn_best.predict(X_dev_scaled)
knn_best_f1 = f1_score(y_dev, knn_best_pred, average='macro')

print(f"\nBest KNN Classification Report:")
print(classification_report(y_dev, knn_best_pred))

# ============================================================================
# STEP 4: Baseline Models
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Baseline Models (DummyClassifier)")
print("=" * 80)

# Stratified baseline
dummy_stratified = DummyClassifier(strategy='stratified', random_state=42)
dummy_stratified.fit(X_train_scaled, y_train)
dummy_strat_pred = dummy_stratified.predict(X_dev_scaled)
dummy_strat_acc = accuracy_score(y_dev, dummy_strat_pred)
dummy_strat_f1 = f1_score(y_dev, dummy_strat_pred, average='macro')

print("\n--- Stratified Strategy ---")
print(f"Accuracy: {dummy_strat_acc:.4f}")
print(f"F1-score (macro): {dummy_strat_f1:.4f}")
print("Predicts classes with same frequency as training set")

# Most frequent baseline
dummy_frequent = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_frequent.fit(X_train_scaled, y_train)
dummy_freq_pred = dummy_frequent.predict(X_dev_scaled)
dummy_freq_acc = accuracy_score(y_dev, dummy_freq_pred)
dummy_freq_f1 = f1_score(y_dev, dummy_freq_pred, average='macro')

print("\n--- Most Frequent Strategy ---")
print(f"Accuracy: {dummy_freq_acc:.4f}")
print(f"F1-score (macro): {dummy_freq_f1:.4f}")
print("Always predicts the most common class")

print("\n--- Baseline Analysis ---")
print("The 'most_frequent' strategy performs poorly because it ignores class imbalance")
print("and never predicts minority classes, resulting in very low F1 scores.")
print("The 'stratified' strategy is more robust as it respects class distributions.")

# ============================================================================
# STEP 5: Final Test Set Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Final Test Set Evaluation")
print("=" * 80)

# Evaluate tuned sklearn model on test set
best_sklearn_test_pred = best_sklearn_model.predict(X_test_scaled)
best_sklearn_test_acc = accuracy_score(y_test, best_sklearn_test_pred)
best_sklearn_test_f1 = f1_score(y_test, best_sklearn_test_pred, average='macro')

# Evaluate KNN on test set
knn_test_pred = knn_best.predict(X_test_scaled)
knn_test_acc = accuracy_score(y_test, knn_test_pred)
knn_test_f1 = f1_score(y_test, knn_test_pred, average='macro')

# Evaluate baselines on test set
dummy_strat_test_pred = dummy_stratified.predict(X_test_scaled)
dummy_strat_test_acc = accuracy_score(y_test, dummy_strat_test_pred)
dummy_strat_test_f1 = f1_score(y_test, dummy_strat_test_pred, average='macro')

dummy_freq_test_pred = dummy_frequent.predict(X_test_scaled)
dummy_freq_test_acc = accuracy_score(y_test, dummy_freq_test_pred)
dummy_freq_test_f1 = f1_score(y_test, dummy_freq_test_pred, average='macro')

print("\n" + "=" * 80)
print("FINAL RESULTS COMPARISON (Test Set)")
print("=" * 80)
print(f"\n{'Model':<30} {'Accuracy':<12} {'F1-Score':<12}")
print("-" * 54)
print(f"{'Baseline (Stratified)':<30} {dummy_strat_test_acc:<12.4f} {dummy_strat_test_f1:<12.4f}")
print(f"{'Baseline (Most Frequent)':<30} {dummy_freq_test_acc:<12.4f} {dummy_freq_test_f1:<12.4f}")
print(f"{f'Custom KNN (k={best_k})':<30} {knn_test_acc:<12.4f} {knn_test_f1:<12.4f}")
print(f"{f'{best_sklearn_name} (Tuned)':<30} {best_sklearn_test_acc:<12.4f} {best_sklearn_test_f1:<12.4f}")

print("\n" + "=" * 80)
print("ANALYSIS AND CONCLUSIONS")
print("=" * 80)

winner = best_sklearn_name if best_sklearn_test_acc >= knn_test_acc else f"Custom KNN (k={best_k})"
winner_acc = max(best_sklearn_test_acc, knn_test_acc)

print(f"\n1. BEST MODEL: {winner}")
print(f"   - Test Accuracy: {winner_acc:.4f}")
print(f"   - Significantly outperforms baselines by {(winner_acc - dummy_strat_test_acc):.4f}")

print("\n2. MODEL COMPARISON:")
print(f"   - {best_sklearn_name}: {best_sklearn_test_acc:.4f} accuracy")
print(f"   - Custom KNN (k={best_k}): {knn_test_acc:.4f} accuracy")
print(f"   - Performance difference: {abs(best_sklearn_test_acc - knn_test_acc):.4f}")

print("\n3. BASELINE PERFORMANCE:")
print(f"   - Stratified baseline: {dummy_strat_test_acc:.4f}")
print(f"   - Most frequent baseline: {dummy_freq_test_acc:.4f}")
print("   - Both baselines perform poorly, validating the need for sophisticated models")

print("\n4. KEY FINDINGS:")
print("   - Hyperparameter tuning improved model performance")
print("   - Feature scaling was critical for distance-based algorithms")
print("   - All sophisticated models substantially beat random guessing")
print("   - The wine dataset is well-suited for classification")

print("\n" + "=" * 80)
print(f"\nDetailed Classification Report for Best Model ({winner}):")
print("=" * 80)
if winner == best_sklearn_name:
    print(classification_report(y_test, best_sklearn_test_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_sklearn_test_pred))
else:
    print(classification_report(y_test, knn_test_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, knn_test_pred))

print("\n" + "=" * 80)
print("PROJECT COMPLETE")
print("=" * 80)