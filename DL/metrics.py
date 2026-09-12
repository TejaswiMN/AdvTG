import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Metrics for transformer models
def transformer_metrics(p):
    """Compute metrics for transformer models."""
    predictions, labels = p
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)

    # sklearn rather than datasets.load_metric, which was removed in datasets 3.0
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def custom_metrics(preds, labels):
    """Compute metrics for custom models using sklearn."""
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1 