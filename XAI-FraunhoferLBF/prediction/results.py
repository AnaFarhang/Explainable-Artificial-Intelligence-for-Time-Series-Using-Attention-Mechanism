from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import torch.nn.functional as F


class results:
    # This class consists of finding attention data and computing performance measurements such as accuracy
    def __init__(self, predictions_probs, predictions, targets, attn, num_classes):

        # Finding attention data
        # For each predicted label, we find the attention data
        label_dict={}
        for i in range(num_classes):
            label_dict[str(i)]=[]
        for j in range(num_classes):
            for i in range(len(predictions.tolist())):
                if predictions.tolist()[i]==j:
                    label_dict[str(j)].append(i)
        self.attn_label_dict={}
        for i in range(num_classes):
            self.attn_label_dict[str(i)]=attn[label_dict[str(i)]]


        # Computing performance measurements
        # Compare predicted labels with the targets
        cm = confusion_matrix(targets.cpu(), predictions.cpu())
        print(cm)
        # Convert logits to probabilities
        predictions_probs = F.softmax(predictions_probs, dim=1)
        # Extract the positive class scores
        predictions_probs_faulty_class = predictions_probs[:, 1].cpu().numpy()
        # Ensure these are NumPy arrays
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        # Accuracy
        self.accuracy = accuracy_score(targets, predictions)
        # Precision (weighted, per-class, or macro-average can be chosen)
        self.precision = precision_score(targets, predictions, average='weighted')
        # Recall
        self.recall = recall_score(targets, predictions, average='weighted')
        # F1 Score
        self.f1 = f1_score(targets, predictions, average='weighted')
        self.auc = roc_auc_score(targets, predictions_probs_faulty_class)
        # Draw ROC
        fpr, tpr, thresholds = roc_curve(targets, predictions_probs_faulty_class)
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()
        # Return rounded results
        self.performance_results = {'accuracy': [round(self.accuracy, 3)], 'precision':[round(self.precision, 3)], 'recall':[round(self.recall, 3)],
                        'f1':[round(self.f1, 3)], 'auc':[round(self.auc, 3)]}


