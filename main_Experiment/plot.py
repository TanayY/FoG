import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()


def plot_roc(y_true, probs, path):
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.plot(fpr, tpr)
    plt.title("ROC")
    plt.savefig(path)
    plt.close()


def plot_pr(y_true, probs, path):
    p, r, _ = precision_recall_curve(y_true, probs)
    plt.plot(r, p)
    plt.title("PR")
    plt.savefig(path)
    plt.close()


def plot_feature_importance(model, path):
    if hasattr(model, "feature_importances_"):
        plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
        plt.title("Feature Importance")
        plt.savefig(path)
        plt.close()


def plot_model_comparison(df, path):
    df.groupby("model")["test_f1"].mean().plot(kind="bar")
    plt.title("Model Comparison (F1)")
    plt.savefig(path)
    plt.close()
