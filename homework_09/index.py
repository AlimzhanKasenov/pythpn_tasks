import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score
)

# ===============================================================
# 1. Загрузка данных
# ===============================================================
df = pd.read_csv("data.csv")

print("\n=== Первые строки ===")
print(df.head(), "\n")

print("Размерность:", df.shape)
print("Количество пропусков:", df.isna().sum().sum())

# ===============================================================
# 2. Предобработка
# ===============================================================
df = df.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df.columns])

df["target"] = df["diagnosis"].map({"M": 1, "B": 0})
X = df.drop(columns=["diagnosis", "target"])
y = df["target"]

print("\nФичей:", X.shape[1])
print("Целевой класс: 1 — злокачественная, 0 — доброкачественная")

# ===============================================================
# 3. EDA (описательная статистика)
# ===============================================================
print("\n=== Базовые статистики ===")
print(df.describe().T)

# --- Корреляция ---
corr = X.corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', aspect='auto')
plt.title("Матрица корреляций (heatmap)")
plt.colorbar()
plt.tight_layout()
plt.show()

# --- Сильно скоррелированные признаки ---
high_corr = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.85:
            high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
print("\nСильно скоррелированные признаки (|r|>0.85):")
for a, b, r in high_corr[:10]:
    print(f"{a:25s} - {b:25s}: r={r:.3f}")

# --- Boxplot ---
for col in X.columns[:6]:
    plt.figure(figsize=(4, 3))
    plt.boxplot([X[y == 0][col], X[y == 1][col]], labels=["B(0)", "M(1)"])
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.show()

# ===============================================================
# 4. Разделение и стандартизация
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nСтандартизация нужна, чтобы признаки с разными масштабами не искажали расстояния в kNN.")

# ===============================================================
# 5. Модель kNN — базовая
# ===============================================================
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
y_proba = knn.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba)
}
print("\n=== kNN (из коробки) ===")
for k, v in metrics.items():
    print(f"{k:10s}: {v:.3f}")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC-кривая kNN")
plt.legend()
plt.show()

# ===============================================================
# 6. Подбор k через кросс-валидацию
# ===============================================================
k_values = range(1, 26)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"\nЛучший k = {best_k} (ROC AUC={max(cv_scores):.3f})")

plt.plot(k_values, cv_scores, marker='o')
plt.title("Подбор k (ROC AUC на CV)")
plt.xlabel("k")
plt.ylabel("ROC AUC")
plt.show()

# Итоговая модель с лучшим k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_best = knn_best.predict(X_test_scaled)
y_proba_best = knn_best.predict_proba(X_test_scaled)[:, 1]

print("\n=== kNN с оптимальным k ===")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("ROC AUC :", roc_auc_score(y_test, y_proba_best))

# ===============================================================
# 7. Бонус — Логистическая регрессия
# ===============================================================
corr_abs = X.corr().abs()
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.85)]
X_red = X.drop(columns=to_drop)
print(f"\nУдалено {len(to_drop)} скоррелированных признаков (>0.85)")

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_red, y, test_size=0.3, random_state=42, stratify=y
)

scaler_r = StandardScaler()
Xr_train_scaled = scaler_r.fit_transform(Xr_train)
Xr_test_scaled = scaler_r.transform(Xr_test)

logreg = LogisticRegression(max_iter=500, solver="liblinear")
logreg.fit(Xr_train_scaled, yr_train)
yr_proba = logreg.predict_proba(Xr_test_scaled)[:, 1]

print("\n=== Logistic Regression (базовая) ===")
print("Accuracy:", accuracy_score(yr_test, logreg.predict(Xr_test_scaled)))
print("ROC AUC :", roc_auc_score(yr_test, yr_proba))

fpr, tpr, _ = roc_curve(yr_test, yr_proba)
plt.plot(fpr, tpr, label="LogReg base")
plt.plot([0, 1], [0, 1], '--')
plt.title("ROC-кривая Logistic Regression")
plt.legend()
plt.show()

# --- Подбор регуляризации C ---
logreg_cv = LogisticRegressionCV(cv=5, solver="liblinear", scoring="roc_auc", max_iter=500)
logreg_cv.fit(Xr_train_scaled, yr_train)
yr_proba_cv = logreg_cv.predict_proba(Xr_test_scaled)[:, 1]

print("\n=== LogisticRegressionCV ===")
print("Accuracy:", accuracy_score(yr_test, logreg_cv.predict(Xr_test_scaled)))
print("ROC AUC :", roc_auc_score(yr_test, yr_proba_cv))

# --- Сравнение ---
print("\n=== Сравнение моделей ===")
print(f"kNN (k={best_k}) ROC AUC: {roc_auc_score(y_test, y_proba_best):.3f}")
print(f"LogisticRegressionCV ROC AUC: {roc_auc_score(yr_test, yr_proba_cv):.3f}")
