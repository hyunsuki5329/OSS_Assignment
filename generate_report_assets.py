import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest (100 trees)": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "Random Forest (300 trees)": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
        ),
        "Random Forest (depth=5)": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
        ),
    }


def save_dataset_image(X, y, target_names, out_dir):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter = axes[0].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap="coolwarm",
        alpha=0.7,
        s=20,
    )
    axes[0].set_title("Dataset image (PCA 2D)")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    legend = axes[0].legend(*scatter.legend_elements(), title="Class")
    axes[0].add_artist(legend)

    class_counts = np.bincount(y)
    axes[1].bar(target_names, class_counts, color=["#d95f5f", "#5f8dd9"])
    axes[1].set_title("Class distribution")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(out_dir / "dataset_image.png", dpi=150)
    plt.close(fig)


def save_mask_image(y, out_dir):
    # Tabular data has no segmentation mask, so label map is shown as mask-like visualization.
    sample_count = 400
    label_map = y[:sample_count].reshape(20, 20)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(label_map, cmap="viridis", interpolation="nearest")
    ax.set_title("Mask-like label map (20x20)")
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "label_mask_image.png", dpi=150)
    plt.close(fig)


def save_prediction_image(y_true, y_pred, out_dir):
    sample_n = 30
    comparison = np.vstack([y_true[:sample_n], y_pred[:sample_n]])

    fig, ax = plt.subplots(figsize=(12, 2.8))
    im = ax.imshow(comparison, cmap="coolwarm", aspect="auto")
    ax.set_title("Prediction image (top: actual, bottom: predicted)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual", "Predicted"])
    ax.set_xlabel("Test sample index")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_image.png", dpi=150)
    plt.close(fig)


def save_accuracy_graph(results, out_dir):
    model_names = list(results.keys())
    scores = [results[name]["accuracy"] for name in model_names]
    x_pos = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x_pos, scores, color=["#507dbc", "#6aa84f", "#f6b26b", "#c27ba0"])
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy graph by model")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha="right")

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.001, f"{score:.4f}", ha="center")

    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_graph.png", dpi=150)
    plt.close(fig)


def save_confusion_matrix(best_model, X_test, y_test, out_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix (best model)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_roc_curve(best_model, X_test, y_test, out_dir):
    if hasattr(best_model, "predict_proba"):
        y_score = best_model.predict_proba(X_test)[:, 1]
    else:
        y_score = best_model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_score = roc_auc_score(y_test, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color="#1f77b4")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve (best model)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    return auc_score


def main():
    output_dir = Path("report_assets")
    output_dir.mkdir(exist_ok=True)

    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = build_models()
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=list(dataset.target_names),
                output_dict=True,
            ),
        }

    best_model_name = max(results, key=lambda key: results[key]["accuracy"])
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    best_pred = best_model.predict(X_test)

    save_dataset_image(X, y, dataset.target_names, output_dir)
    save_mask_image(y, output_dir)
    save_prediction_image(y_test, best_pred, output_dir)
    save_accuracy_graph(results, output_dir)
    save_confusion_matrix(best_model, X_test, y_test, output_dir)
    auc_score = save_roc_curve(best_model, X_test, y_test, output_dir)

    metrics = {
        "dataset": {
            "name": "load_breast_cancer",
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "class_names": list(dataset.target_names),
            "class_counts": {
                dataset.target_names[index]: int(count)
                for index, count in enumerate(np.bincount(y))
            },
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
        },
        "model_results": {
            model_name: {
                "accuracy": round(result["accuracy"], 6),
                "precision_macro": round(result["classification_report"]["macro avg"]["precision"], 6),
                "recall_macro": round(result["classification_report"]["macro avg"]["recall"], 6),
                "f1_macro": round(result["classification_report"]["macro avg"]["f1-score"], 6),
            }
            for model_name, result in results.items()
        },
        "best_model": {
            "name": best_model_name,
            "accuracy": round(results[best_model_name]["accuracy"], 6),
            "roc_auc": round(float(auc_score), 6),
        },
        "generated_images": [
            "report_assets/dataset_image.png",
            "report_assets/label_mask_image.png",
            "report_assets/prediction_image.png",
            "report_assets/accuracy_graph.png",
            "report_assets/confusion_matrix.png",
            "report_assets/roc_curve.png",
        ],
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"Report assets generated in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()