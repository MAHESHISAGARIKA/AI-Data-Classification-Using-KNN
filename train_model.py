import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


def train_and_save_model():
    # Create folders if they do not exist
    os.makedirs("model", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Load Iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    target_names = iris.target_names

    # Convert dataset into DataFrame for better understanding
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    df["species"] = df["target"].apply(lambda x: target_names[x])

    print("First 5 rows of dataset:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nClass distribution:")
    print(df["species"].value_counts())

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create ML pipeline
    # StandardScaler balances feature values
    # KNN classifies based on nearest neighbors
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print("\nModel Accuracy:")
    print(f"{accuracy * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    # Save trained model
    model_data = {
        "model": model,
        "target_names": target_names,
        "feature_names": feature_names,
        "accuracy": accuracy
    }

    joblib.dump(model_data, "model/iris_knn_model.pkl")

    # Save confusion matrix image
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )

    display.plot()
    plt.title("Confusion Matrix - Iris KNN Classification")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # Save report into text file
    with open("outputs/model_report.txt", "w") as file:
        file.write("AI Data Classification Using KNN\n")
        file.write("================================\n\n")
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        file.write("Confusion Matrix:\n")
        file.write(str(cm))
        file.write("\n\nClassification Report:\n")
        file.write(report)

    print("\nModel saved successfully in model/iris_knn_model.pkl")
    print("Confusion matrix saved in outputs/confusion_matrix.png")
    print("Model report saved in outputs/model_report.txt")


if __name__ == "__main__":
    train_and_save_model()