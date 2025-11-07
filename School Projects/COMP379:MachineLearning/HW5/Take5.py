# Training Logisitic Regression using n-fold Cross-Validation 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np  

# Load Dataset
url = 'wine/wine.data'
data = pd.read_csv(url)

# Separate features and target
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

print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

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
            X_fold = X[fold_indices]
            Y_fold = Y[fold_indices]
            # Append fold
            folds.append((X_fold, Y_fold))

        return folds
        

    def split(self, folds, val_fold_index):
        # Split data into training and validation sets
        X_val, Y_val = folds[val_fold_index]
        # Initialize training data lists
        X_train_list = []
        Y_train_list = []
        # Combine other folds into training set
        for i, (X_fold, Y_fold) in enumerate(folds):
            # Add fold to training set if not validation fold
            if i != val_fold_index:
                # Append fold data to training lists
                X_train_list.append(X_fold)
                Y_train_list.append(Y_fold)
        # Stack training data
        X_train = np.vstack(X_train_list)
        Y_train = np.concatenate(Y_train_list)

        return X_train, Y_train, X_val, Y_val
    
    def train(self, model, X, Y, **model_params):
        # Train model using n-fold CV
        self.fold_scores = []
        self.fold_models = []
        # Create randomized folds
        folds = self.randomize_folds(X, Y)
        # Train on each fold
        for fold_index in range(self.n_folds):
            print(f"\nTraining fold {fold_index + 1}/{self.n_folds}")
            # Split data
            X_train, Y_train, X_val, Y_val = self.split(folds, fold_index)
            # Initialize model
            model_instance = model(**model_params)
            # Train model
            model_instance.fit(X_train, Y_train)
            # Make predictions
            Y_pred = model_instance.predict(X_val)
            # Calculate accuracy
            acc = accuracy_score(Y_val, Y_pred)
            # Store fold results
            self.fold_scores.append({'fold': fold_index, 'accuracy': acc})
            self.fold_models.append(model_instance)

            print(f"Accuracy: {acc:.4f}")
        
        # Calc stats
        accuracies = [s['accuracy'] for s in self.fold_scores]
        avg_acc = np.mean(accuracies)

        return  {'average_accuracy': avg_acc}
    
    
    
cv = NFoldCrossValidation(n_folds=5)

cv_results = cv.train(
    LogisticRegression,
    X_train_scaled,
    Y_train,
    max_iter=10000,
    random_state=42
)

lr_model = LogisticRegression(max_iter=10000, random_state=42)
lr_model.fit(X_train_scaled, Y_train)

Y_pred_test = lr_model.predict(X_test_scaled)
test_acc = accuracy_score(Y_test, Y_pred_test)

Y_pred_train = lr_model.predict(X_train_scaled)
train_acc = accuracy_score(Y_train, Y_pred_train)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"CV Average Accuracy: {cv_results['average_accuracy']:.4f}")

