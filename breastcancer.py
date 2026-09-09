# Breast Cancer Prediction using Machine Learning and Deep Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

# ============================================================
# 2. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 3. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. MACHINE LEARNING MODEL - LOGISTIC REGRESSION
# ============================================================

ml_model = LogisticRegression(max_iter=5000)

ml_model.fit(X_train_scaled, y_train)

ml_pred = ml_model.predict(X_test_scaled)

ml_accuracy = accuracy_score(y_test, ml_pred)

print("\n========================================")
print("MACHINE LEARNING - LOGISTIC REGRESSION")
print("========================================")
print("Accuracy:", ml_accuracy)

# ============================================================
# 5. DEEP LEARNING MODEL - ARTIFICIAL NEURAL NETWORK
# ============================================================

print("\n========================================")
print("DEEP LEARNING - ARTIFICIAL NEURAL NETWORK")
print("========================================")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 6. EARLY STOPPING
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ============================================================
# 7. TRAIN DEEP LEARNING MODEL
# ============================================================

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

dl_probabilities = model.predict(X_test_scaled)

dl_pred = (dl_probabilities >= 0.5).astype(int).flatten()

# ============================================================
# 9. EVALUATE DEEP LEARNING MODEL
# ============================================================

dl_accuracy = accuracy_score(y_test, dl_pred)
precision = precision_score(y_test, dl_pred)
recall = recall_score(y_test, dl_pred)
f1 = f1_score(y_test, dl_pred)

print("\nDeep Learning Model Results")
print("----------------------------------------")
print("Accuracy :", dl_accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report:")
print(classification_report(
    y_test,
    dl_pred,
    target_names=data.target_names
))

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, dl_pred)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Deep Learning Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix_dl.png")
plt.show()

# ============================================================
# 11. TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ANN Training and Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_accuracy.png")
plt.show()

# ============================================================
# 12. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Training and Validation Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_loss.png")
plt.show()

# ============================================================
# 13. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, dl_probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ANN (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Deep Learning Model")
plt.legend()

plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print("\nROC-AUC:", roc_auc)

# ============================================================
# 14. MODEL COMPARISON
# ============================================================

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")
print(f"Logistic Regression Accuracy: {ml_accuracy:.4f}")
print(f"ANN Accuracy:                 {dl_accuracy:.4f}")

if dl_accuracy > ml_accuracy:
    print("ANN performed better than Logistic Regression.")
elif dl_accuracy < ml_accuracy:
    print("Logistic Regression performed better than ANN.")
else:
    print("Both models achieved the same accuracy.")

# ============================================================
# 15. SAVE DEEP LEARNING MODEL
# ============================================================

model.save("breast_cancer_ann.keras")

print("\nDeep Learning model saved as:")
print("breast_cancer_ann.keras")

print("\nProject execution completed successfully!")
