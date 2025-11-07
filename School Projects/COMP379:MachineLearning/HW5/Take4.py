# ============================================
# HW 5 - Part 1: Train and Evaluate Classifier using N-Fold Cross-Validation
# ============================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np  

# ============================================
# Load and Prepare Dataset
# ============================================

print("="*70)
print("DATASET PREPARATION")
print("="*70)

# Load Dataset
url = 'wine/wine.data'
data = pd.read_csv(url)

# Separate features and target
X = data.drop('Class', axis=1).values
Y = data['Class'].values    

print(f"Total samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes: {sorted(np.unique(Y))}")
print(f"Class distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")

# Split Data -> 80% Train, 20% Test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Standardization (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# ============================================
# N-Fold Cross-Validation Implementation
# ============================================

class NFoldCrossValidation:
    """
    Custom N-Fold Cross-Validation implementation.
    
    Splits training data into n folds and trains model n times,
    using each fold as validation set once while training on the rest.
    """
    
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.fold_scores = []
        self.fold_models = []
    
    def split(self, X, Y, fold_index):
        """
        Split data into training and validation sets for given fold.
        
        Parameters:
        -----------
        X : numpy array, feature data
        Y : numpy array, target labels
        fold_index : int, which fold to use as validation (0 to n_folds-1)
        
        Returns:
        --------
        X_train, Y_train, X_val, Y_val
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_folds
        
        # Determine validation indices
        val_start = fold_index * fold_size
        # Last fold takes the remainder to include all samples
        val_end = (fold_index + 1) * fold_size if fold_index < self.n_folds - 1 else n_samples
        
        # Create indexes for training and validation sets
        val_indices = np.arange(val_start, val_end)
        train_indices = np.concatenate((
            np.arange(0, val_start), 
            np.arange(val_end, n_samples)
        ))
        
        # Split the data
        X_train, Y_train = X[train_indices], Y[train_indices]
        X_val, Y_val = X[val_indices], Y[val_indices]
        
        return X_train, Y_train, X_val, Y_val
    
    def train(self, model, X, Y, **model_params):
        """
        Train model using n-fold cross-validation.
        
        Parameters:
        -----------
        model : class, model class to instantiate (e.g., LogisticRegression)
        X : numpy array, feature data
        Y : numpy array, target labels
        **model_params : keyword arguments to pass to model constructor
        
        Returns:
        --------
        dict : results containing fold scores and average accuracy
        """
        print("\n" + "="*70)
        print(f"N-FOLD CROSS-VALIDATION (n={self.n_folds})")
        print("="*70)
        print(f"Total training samples: {len(X)}")
        print(f"Approximate samples per fold: {len(X) // self.n_folds}")
        print("-"*70)
        
        # Reset results
        self.fold_scores = []
        self.fold_models = []
        
        # Train on each fold
        for fold_index in range(self.n_folds):
            print(f"\nFold {fold_index + 1}/{self.n_folds}:")
            
            # Split data
            X_train, Y_train, X_val, Y_val = self.split(X, Y, fold_index)
            
            print(f"  Training samples: {len(X_train)}")
            print(f"  Validation samples: {len(X_val)}")
            
            # Train model
            model_instance = model(**model_params)
            model_instance.fit(X_train, Y_train)
            
            # Make predictions
            Y_pred = model_instance.predict(X_val)
            
            # Calculate accuracy
            acc = accuracy_score(Y_val, Y_pred)
            
            # Store results
            fold_result = {
                'fold': fold_index + 1,
                'accuracy': acc,
                'train_size': len(X_train),
                'val_size': len(X_val)
            }
            
            self.fold_scores.append(fold_result)
            self.fold_models.append(model_instance)
            
            print(f"  Accuracy: {acc:.4f}")
        
        # Calculate statistics
        accuracies = [s['accuracy'] for s in self.fold_scores]
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        
        print("\n" + "-"*70)
        print("CROSS-VALIDATION SUMMARY:")
        print(f"  Average Accuracy: {avg_acc:.4f}")
        print(f"  Std Deviation:    {std_acc:.4f}")
        print(f"  Min Accuracy:     {min_acc:.4f}")
        print(f"  Max Accuracy:     {max_acc:.4f}")
        print("-"*70)
        
        return {
            'fold_scores': self.fold_scores,
            'fold_models': self.fold_models,
            'average_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'min_accuracy': min_acc,
            'max_accuracy': max_acc
        }
    
    def get_best_model(self):
        """Return the model from the fold with highest accuracy."""
        best_index = np.argmax([s['accuracy'] for s in self.fold_scores])
        return self.fold_models[best_index], self.fold_scores[best_index]


# ============================================
# Train Using N-Fold Cross-Validation
# ============================================

# Create cross-validation object
cv = NFoldCrossValidation(n_folds=5)

# Train and evaluate using cross-validation
cv_results = cv.train(
    LogisticRegression,
    X_train_scaled,
    Y_train,
    max_iter=10000,
    random_state=42
)

# ============================================
# Train Final Model on Full Training Set
# ============================================

print("\n" + "="*70)
print("FINAL MODEL EVALUATION")
print("="*70)

# Train logistic regression on entire training set
lr_model = LogisticRegression(max_iter=10000, random_state=42)
lr_model.fit(X_train_scaled, Y_train)

# Evaluate on test set
Y_pred_test = lr_model.predict(X_test_scaled)
test_acc = accuracy_score(Y_test, Y_pred_test)

# Evaluate on training set (to check for overfitting)
Y_pred_train = lr_model.predict(X_train_scaled)
train_acc = accuracy_score(Y_train, Y_pred_train)

print(f"\nTraining Set Accuracy:   {train_acc:.4f}")
print(f"Test Set Accuracy:       {test_acc:.4f}")
print(f"CV Average Accuracy:     {cv_results['average_accuracy']:.4f}")

# ============================================
# Performance Analysis
# ============================================

print("\n" + "="*70)
print("PERFORMANCE ANALYSIS")
print("="*70)

# Compare training vs test accuracy
train_test_diff = train_acc - test_acc
print(f"\nTraining vs Test Difference: {train_test_diff:.4f} ({train_test_diff*100:.2f}%)")

if train_test_diff > 0.05:
    print("⚠️  Model shows signs of overfitting (training accuracy >> test accuracy)")
elif train_test_diff > 0.02:
    print("✓  Minor overfitting detected, but model generalizes reasonably well")
else:
    print("✓✓ Excellent generalization - model performs similarly on training and test data")

# Compare CV estimate vs actual test performance
cv_test_diff = abs(cv_results['average_accuracy'] - test_acc)
print(f"\nCV Estimate vs Test Difference: {cv_test_diff:.4f} ({cv_test_diff*100:.2f}%)")

if cv_test_diff < 0.02:
    print("✓✓ Cross-validation provided an accurate estimate of test performance")
elif cv_test_diff < 0.05:
    print("✓  Cross-validation provided a reasonable estimate of test performance")
else:
    print("⚠️  Significant gap between CV estimate and test performance")

# Stability analysis
print(f"\nModel Stability (CV Std Dev): {cv_results['std_accuracy']:.4f}")
if cv_results['std_accuracy'] < 0.02:
    print("✓✓ Very stable - consistent performance across folds")
elif cv_results['std_accuracy'] < 0.05:
    print("✓  Reasonably stable performance")
else:
    print("⚠️  High variance across folds - model may be sensitive to data splits")

# ============================================
# Summary for Homework Writeup
# ============================================

print("\n" + "="*70)
print("SUMMARY FOR HOMEWORK WRITEUP")
print("="*70)

print(f"""
1. Model: Logistic Regression
2. Cross-Validation: {cv.n_folds}-fold
3. Dataset: Wine (stratified 80/20 train-test split)
4. Preprocessing: StandardScaler (mean=0, std=1)

Results:
- CV Average Accuracy: {cv_results['average_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}
- Training Accuracy:    {train_acc:.4f}
- Test Accuracy:        {test_acc:.4f}

Key Findings:
- Cross-validation provided {'an accurate' if cv_test_diff < 0.02 else 'a reasonable'} estimate of test performance
- Model shows {'minimal' if train_test_diff < 0.02 else 'some'} overfitting
- Performance is {'very stable' if cv_results['std_accuracy'] < 0.02 else 'reasonably stable'} across folds
- The classifier generalizes {'excellently' if test_acc > 0.95 else 'well'} to unseen data

Interpretation:
The n-fold cross-validation approach provides a more reliable estimate of model
performance compared to a single train-test split. By training on {cv.n_folds} different
subsets and averaging the results, we reduce the variance in our performance
estimate and get a better sense of how the model will perform on new data.
""")