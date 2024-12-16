Here are comprehensive notes on Support Vector Machines (SVM):

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

Copy

Insert at cursor
python
Key Concepts of SVM:

Basic Principle :

# SVM finds the optimal hyperplane that maximizes the margin between classes
svm = SVC(kernel='linear')  # Linear SVM
# For non-linear separation:
svm_rbf = SVC(kernel='rbf')  # RBF (Radial Basis Function) kernel

Copy

Insert at cursor
python
Important Parameters :

svm = SVC(
    C=1.0,              # Regularization parameter
    kernel='rbf',       # Kernel type: 'linear', 'rbf', 'poly', 'sigmoid'
    gamma='scale',      # Kernel coefficient
    class_weight=None,  # Handle imbalanced classes
    probability=True    # Enable probability estimates
)

Copy

Insert at cursor
python
Common Kernels :

# Linear Kernel - for linearly separable data
svm_linear = SVC(kernel='linear')

# RBF Kernel - for non-linear data
svm_rbf = SVC(kernel='rbf')

# Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3)

Copy

Insert at cursor
python
Complete Implementation Example :

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y, 
    test_size=0.2, 
    random_state=42
)

# Create and train model
svm_classifier = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)

# Train
svm_classifier.fit(X_train, y_train)

# Predict
y_pred = svm_classifier.predict(X_test)

# Get probabilities
y_prob = svm_classifier.predict_proba(X_test)

# Evaluate
from sklearn.metrics import classification_report, accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

Copy

Insert at cursor
python
Hyperparameter Tuning :

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)

Copy

Insert at cursor
python
Key Points to Remember:

Advantages :

Effective in high-dimensional spaces

Memory efficient

Versatile (different kernel functions)

Works well with clear margin of separation

Disadvantages :

Not suitable for large datasets (training time O(n²))

Sensitive to feature scaling

Requires careful parameter tuning

Less effective when classes overlap significantly

Best Practices :

# Always scale your features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalanced classes
svm_imbalanced = SVC(
    class_weight='balanced',
    random_state=42
)

# Cross-validation for reliable performance estimation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svm_classifier, X_scaled, y, cv=5)
print("Cross-validation scores:", scores.mean())

Copy

Insert at cursor
python
When to Use SVM :

Medium-sized datasets

Complex decision boundaries needed

When data has clear margin of separation

Binary or multiclass classification

Feature Selection with SVM :

from sklearn.feature_selection import SelectFromModel

# Use linear SVM for feature selection
svm_selector = SVC(kernel='linear')
selector = SelectFromModel(svm_selector)
X_selected = selector.fit_transform(X_scaled, y)

Copy

Insert at cursor
python
Remember:

Always scale your features before using SVM

Start with linear kernel, then try RBF if needed

Use cross-validation for parameter tuning

Consider computational cost for large datasets

Monitor for overfitting, especially with RBF kernel

