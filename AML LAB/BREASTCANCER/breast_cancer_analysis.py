import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_breast_cancer()
X, y = data.data, data.target

print("=== Dataset Info ===")
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Feature names: {list(data.feature_names)}")
print(f"Classes: {list(data.target_names)}")
print(f"Class counts — Malignant(0): {(y==0).sum()}, Benign(1): {(y==1).sum()}\n")

plt.figure(figsize=(5, 4))
sns.countplot(x=y, palette="Set2")
plt.xticks([0, 1], data.target_names)
plt.title("Class Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=100)
plt.show()
print("Class distribution plot saved.\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== Evaluation ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report (Precision & Recall):")
print(classification_report(y_test, y_pred, target_names=data.target_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.show()
print("Confusion matrix plot saved.\n")

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1][:10]   # top-10

plt.figure(figsize=(9, 5))
sns.barplot(x=importances[indices], y=np.array(data.feature_names)[indices],
            palette="viridis")
plt.title("Top-10 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100)
plt.show()
print("Top-10 Feature Importances:")
for rank, i in enumerate(indices, 1):
    print(f"  {rank:2d}. {data.feature_names[i]:<35s} {importances[i]:.4f}")

print("""
WHY these features matter:
  • 'worst radius / worst perimeter / worst area'  — larger tumour size is strongly
    associated with malignancy; the 'worst' (largest) cell measurements capture the
    most extreme cells in a sample, which are most indicative of aggressive tumours.
  • 'worst concave points / mean concave points'   — concave portions of the cell
    contour reflect irregular, jagged boundaries typical of malignant cells.
  • 'worst concavity'                              — complements concave-points by
    measuring the severity of those concavities.
  Random Forest's Gini-impurity-based importance ranks features by how much they
  reduce uncertainty across all 100 trees, making it a reliable guide to which
  measurements are truly discriminative.
""")
