import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# TODO: CHANGE URL TO LOCATION OF WINE.DATA FILE
# Load the dataset
url = 'wine.data'
data = pd.read_csv(url)

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# 70% train, 15% dev, 15% test
# Split off training set, then split remaining into development and test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
# Split the temp set (30%) into development and test 
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nTraining set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"Development set size: {len(X_dev)} ({len(X_dev)/len(X)*100:.1f}%)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)
X_test_scaled = scaler.transform(X_test)

# Use either logistic regression or SVM classifiers from scikit-learn 
# to train a classifier. Use the default classifier hyperparameters. 
# Evaluate your classifier on the development set.

# Logistic Regression on Development Set
lr_model = LogisticRegression(max_iter=10000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_dev_scaled)
lr_acc = accuracy_score(y_dev, lr_pred)

print("\n--- Logistic Regression ---")
print(f"Accuracy: {lr_acc:.3f}")

# Now explore the classifier hyperparameters and see if you 
# can improve your model’s performance on the development set.

# Improving Logistic Regression
print("\n--- Improving Logistic Regression ---")

lr_tuned_model = LogisticRegression(max_iter=10000, random_state=42, C=0.1, penalty='l2')
lr_tuned_model.fit(X_train_scaled, y_train)
lr_tuned_pred = lr_tuned_model.predict(X_dev_scaled)
lr_tuned_acc = accuracy_score(y_dev, lr_tuned_pred)

print(f"Accuracy: {lr_tuned_acc:.3f}")

# To establish the performance of a baseline system, apply the Dummy 
# Classifier from scikit-learn (sklearn.dummy.DummyClassifier) to 
# your data. At minimum, try the following two values for the 
# ‘strategy’ parameter: ‘stratified’ and ‘most_frequent’. 
# Comment on your findings.

# stratified parameter
dummy_stratified = DummyClassifier(strategy='stratified', random_state=42)
dummy_stratified.fit(X_train_scaled, y_train)
dummy_strat_pred = dummy_stratified.predict(X_dev_scaled)
dummy_strat_acc = accuracy_score(y_dev, dummy_strat_pred)

print("\n--- stratified Parameter ---")
print(f"Accuracy: {dummy_strat_acc:.3f}")

# most_frequent parameter
dummy_frequent = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_frequent.fit(X_train_scaled, y_train)
dummy_freq_pred = dummy_frequent.predict(X_dev_scaled)
dummy_freq_acc = accuracy_score(y_dev, dummy_freq_pred)

print("\n--- most_frequent Parameter ---")
print(f"Accuracy: {dummy_freq_acc:.3f}")

# Logistic Regression on test set
lr_final_pred = lr_tuned_model.predict(X_test_scaled)
lr_final_acc = accuracy_score(y_test, lr_final_pred)
print("\n--- Logistic Regression on Test Set ---")
print(f"Accuracy: {lr_final_acc:.3f}")

