"""Simple customer churn prediction with NumPy, Pandas, Seaborn, and Matplotlib."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RANDOM_SEED = 42
FEATURES = ["tenure_months", "monthly_charges", "support_calls", "contract_length"]


def create_customer_data(number_of_customers: int = 800) -> pd.DataFrame:
    """Create a realistic, reproducible toy churn dataset."""
    generator = np.random.default_rng(RANDOM_SEED)

    tenure_months = generator.integers(1, 73, number_of_customers)
    monthly_charges = np.clip(
        generator.normal(75, 22, number_of_customers), 25, 160
    ).round(2)
    support_calls = generator.poisson(2.2, number_of_customers)
    contract_length = generator.choice(
        [1, 12, 24], number_of_customers, p=[0.48, 0.34, 0.18]
    )

    # These weights create a learnable relationship while retaining noise.
    churn_score = (
        1.8
        - 0.045 * tenure_months
        + 0.018 * monthly_charges
        + 0.38 * support_calls
        - 0.055 * contract_length
        + generator.normal(0, 0.8, number_of_customers)
    )
    churn_probability = 1 / (1 + np.exp(-churn_score))
    churned = generator.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "support_calls": support_calls,
            "contract_length": contract_length,
            "churned": churned,
        }
    )


def train_test_split(
    data: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data while keeping similar churn rates in both partitions."""
    generator = np.random.default_rng(RANDOM_SEED)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in data.groupby("churned"):
        indices = group.index.to_numpy(copy=True)
        generator.shuffle(indices)
        split_index = int(len(indices) * (1 - test_size))
        train_indices.extend(indices[:split_index])
        test_indices.extend(indices[split_index:])

    return data.loc[train_indices].sample(frac=1, random_state=RANDOM_SEED), data.loc[
        test_indices
    ].sample(frac=1, random_state=RANDOM_SEED)


def standardize(
    train_features: np.ndarray, test_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    means = train_features.mean(axis=0)
    standard_deviations = train_features.std(axis=0)
    return (
        (train_features - means) / standard_deviations,
        (test_features - means) / standard_deviations,
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -500, 500)))


def fit_logistic_regression(
    features: np.ndarray,
    targets: np.ndarray,
    learning_rate: float = 0.08,
    epochs: int = 2500,
) -> tuple[np.ndarray, np.ndarray]:
    """Train logistic regression using only NumPy gradient descent."""
    weights = np.zeros(features.shape[1])
    bias = 0.0

    for _ in range(epochs):
        probabilities = sigmoid(features @ weights + bias)
        errors = probabilities - targets
        weights -= learning_rate * (features.T @ errors) / len(targets)
        bias -= learning_rate * errors.mean()

    return weights, np.array([bias])


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    true_negative = int(((actual == 0) & (predicted == 0)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())
    accuracy = (true_positive + true_negative) / len(actual)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def create_visuals(data: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray) -> Path:
    output_path = Path("output")
    output_path.mkdir(exist_ok=True)
    chart_path = output_path / "churn_analysis.png"

    plot_data = data.copy()
    plot_data["churn_label"] = plot_data["churned"].map({0: "Stayed", 1: "Churned"})
    confusion_matrix = pd.crosstab(
        pd.Series(actual, name="Actual"),
        pd.Series(predicted, name="Predicted"),
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    sns.set_theme(style="whitegrid", palette="deep")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(
        data=plot_data,
        x="tenure_months",
        y="monthly_charges",
        hue="churn_label",
        alpha=0.65,
        ax=axes[0],
    )
    axes[0].set_title("Customer profile and churn")
    axes[0].set_xlabel("Tenure (months)")
    axes[0].set_ylabel("Monthly charges")

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_title("Confusion matrix")
    axes[1].set_xlabel("Predicted label (0 = stay, 1 = churn)")
    axes[1].set_ylabel("Actual label")
    figure.tight_layout()
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    return chart_path


def main() -> None:
    data = create_customer_data()
    train_data, test_data = train_test_split(data)
    train_features, test_features = standardize(
        train_data[FEATURES].to_numpy(dtype=float),
        test_data[FEATURES].to_numpy(dtype=float),
    )
    train_targets = train_data["churned"].to_numpy(dtype=float)
    test_targets = test_data["churned"].to_numpy(dtype=int)

    weights, bias = fit_logistic_regression(train_features, train_targets)
    churn_probabilities = sigmoid(test_features @ weights + bias[0])
    predictions = (churn_probabilities >= 0.5).astype(int)
    metrics = evaluate(test_targets, predictions)
    chart_path = create_visuals(test_data, test_targets, predictions)

    print(f"Customers: {len(data)} | Train: {len(train_data)} | Test: {len(test_data)}")
    print(f"Overall churn rate: {data['churned'].mean():.1%}")
    print("\nTest metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"\nSaved chart to {chart_path}")

    example_customer = np.array([[4, 120, 6, 1]], dtype=float)
    example_customer, _ = standardize(
        train_data[FEATURES].to_numpy(dtype=float), example_customer
    )
    example_probability = sigmoid(example_customer @ weights + bias[0])[0]
    print(
        f"Example customer churn probability: {example_probability:.1%} "
        f"({'likely to churn' if example_probability >= 0.5 else 'likely to stay'})"
    )


if __name__ == "__main__":
    main()