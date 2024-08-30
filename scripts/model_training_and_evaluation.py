import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import optuna
import torch
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt

# Load the dataset with embeddings (ensure embeddings are stored as numpy arrays in the dataframe)
train_set = pd.read_csv('data/train_data.csv')
val_set = pd.read_csv('data/val_data.csv')
test_set = pd.read_csv('data/test_data.csv')

X_train = np.vstack(train_set['embeddings'].apply(eval))
y_train = train_set['Category']
X_val = np.vstack(val_set['embeddings'].apply(eval))
y_val = val_set['Category']
X_test = np.vstack(test_set['embeddings'].apply(eval))
y_test = test_set['Category']

# Define objective function for Optuna
def objective(trial):
    model_name = trial.suggest_categorical('model', ['SVM', 'RandomForest', 'MLP'])

    if model_name == 'SVM':
        C = trial.suggest_float('C', 0.1, 10.0)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        model = SVC(C=C, kernel=kernel, probability=True)
    elif model_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    else:
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50, 50), (100, 50), (100, 100)])
        learning_rate_init = trial.suggest_float('learning_rate_init', 0.001, 0.1)
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, max_iter=300)

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)

    return accuracy

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Train the best model on the full training data
best_model_name = study.best_params['model']
if best_model_name == 'SVM':
    best_model = SVC(C=study.best_params['C'], kernel=study.best_params['kernel'], probability=True)
elif best_model_name == 'RandomForest':
    best_model = RandomForestClassifier(n_estimators=study.best_params['n_estimators'], max_depth=study.best_params['max_depth'])
else:
    best_model = MLPClassifier(hidden_layer_sizes=study.best_params['hidden_layer_sizes'],
                               learning_rate_init=study.best_params['learning_rate_init'], max_iter=300)

best_model.fit(X_train, y_train)

# Evaluate on the test set
test_preds = best_model.predict(X_test)
accuracy = accuracy_score(y_test, test_preds)
precision = precision_score(y_test, test_preds, average='weighted')
recall = recall_score(y_test, test_preds, average='weighted')
f1 = f1_score(y_test, test_preds, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(classification_report(y_test, test_preds))

# Visualize Optuna results
fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)
fig1.show()
fig2.show()

# Save figures
fig1.write_image("optuna_optimization_history.png")
fig2.write_image("optuna_param_importances.png")
