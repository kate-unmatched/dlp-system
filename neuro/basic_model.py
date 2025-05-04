from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Загрузка датасета
df = pd.read_csv("synthetic_behavioral_dataset.csv")
X = df.drop(columns=["danger_score"])
y = df["danger_score"]

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Добавление оси канала для CNN: (samples, features, 1)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


# Создание модели
def create_model(filters=64, kernel_size=3, dropout_rate=0.2, dense_units=128):
    model = models.Sequential([
        layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same',
                      input_shape=(X_train_scaled.shape[1], 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(7, activation='softmax')
    ])
    return model


# Обёртка модели с параметрами для GridSearchCV
model = KerasClassifier(
    model=create_model,
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
    verbose=0,
    filters=64,
    kernel_size=3,
    dropout_rate=0.2,
    dense_units=128
)

# Параметры для перебора
param_grid = {
    "filters": [32, 64],
    "kernel_size": [3, 5],
    "dropout_rate": [0.2, 0.3],
    "dense_units": [128, 256],
    "batch_size": [32],
    "epochs": [10]
}

# GridSearch
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Лучшие параметры и метрики
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# Оценка на тестовой выборке
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print("\nClassification report on test set:\n")
print(classification_report(y_test, y_pred))

# Графики обучения
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(best_model.history_["loss"], label="Train Loss")
plt.plot(best_model.history_["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_model.history_["accuracy"], label="Train Acc")
plt.plot(best_model.history_["val_accuracy"], label="Val Acc")
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
