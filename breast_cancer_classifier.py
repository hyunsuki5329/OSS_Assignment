from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data():
    dataset = load_breast_cancer()
    return dataset.data, dataset.target, dataset.target_names


def split_data(features, labels):
    return train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )


def build_model():
    logistic_regression = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    random_forest_100 = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_300 = RandomForestClassifier(n_estimators=300, random_state=42)
    random_forest_depth_5 = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
    )

    return {
        "Logistic Regression": logistic_regression,
        "Random Forest (100 trees)": random_forest_100,
        "Random Forest (300 trees)": random_forest_300,
        "Random Forest (depth=5)": random_forest_depth_5,
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[model_name] = accuracy_score(y_test, predictions)

    return results


def main():
    features, labels, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(features, labels)

    models = build_model()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    best_model_name, best_accuracy = max(results.items(), key=lambda item: item[1])
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    sample_predictions = best_model.predict(X_test[:10])

    print("Breast cancer classification")
    print(f"Classes: {list(target_names)}")
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    for model_name, accuracy in results.items():
        print(f"{model_name} accuracy: {accuracy:.4f}")
    print(f"Best model: {best_model_name} ({best_accuracy:.4f})")
    print("Sample predictions:")
    for index, predicted_label in enumerate(sample_predictions, start=1):
        actual_label = y_test[index - 1]
        print(
            f"  #{index}: predicted={target_names[predicted_label]}, "
            f"actual={target_names[actual_label]}"
        )


if __name__ == "__main__":
    main()