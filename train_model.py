import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize
import joblib
import os

def main():
    # Load data
    df = pd.read_csv('res.csv')
    X = df.drop('class', axis=1)
    
    # Normalize the histograms (L1 norm makes the sum of each row = 1)
    # This prevents the model from being biased by the image resolution
    X_norm = normalize(X, norm='l1')
    X = pd.DataFrame(X_norm, columns=X.columns)

    y = df['class']

    # Evaluate different algorithms
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': GaussianNB()
    }

    print("Evaluating models using 5-fold cross-validation...")
    best_model_name = None
    best_score = 0

    for name, model in models.items():
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        mean_score = cv_results.mean()
        print(f"{name}: Mean Accuracy = {mean_score:.4f}, Std = {cv_results.std():.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name

    print(f"\nBest model selected: {best_model_name} with accuracy {best_score:.4f}")

    # Train best model on entire dataset
    print(f"Training {best_model_name} on the entire dataset...")
    best_model = models[best_model_name]
    best_model.fit(X, y)

    # Save the model
    joblib.dump(best_model, 'best_model.pkl')
    print("Model saved to best_model.pkl")

if __name__ == '__main__':
    main()
