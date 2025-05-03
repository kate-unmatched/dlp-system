import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

# Загрузка сгенерированного датасета
df = pd.read_csv("balanced_behavioral_data.csv")

# Модификация целевого признака (уменьшаем все метки на 1)
y = df['danger_score']
X = df.drop(columns=['danger_score'])

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ресайзинг данных для CNN (добавление оси канала, чтобы данные подходили для CNN)
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1],
                                        1)  # (samples, features, channels)
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1],
                                      1)  # (samples, features, channels)


# Определение модели CNN
def create_model(filters=64, kernel_size=3, dropout_rate=0.2, dense_units=128):
    model = models.Sequential([
        layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                      input_shape=(X_train_scaled.shape[1], 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(7, activation='softmax')  # 7 классов (от 0 до 6)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Кросс-энтропия для многоклассовой классификации
                  metrics=['accuracy'])

    return model


# Настройка гиперпараметров для GridSearchCV
param_grid = {
    'filters': [32, 64],  # Количество фильтров
    'kernel_size': [3, 5],  # Размер свёрточного ядра
    'dropout_rate': [0.2, 0.3],  # Dropout
    'dense_units': [128, 256],  # Количество нейронов в Dense слое
    'epochs': [10, 20],  # Количество эпох
    'batch_size': [32, 64]  # Размер батча
}

# Используем KerasClassifier для применения GridSearchCV
model = KerasClassifier(model=create_model, verbose=0)

# Запуск GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Печать лучших параметров и результатов
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# Оценка модели с лучшими гиперпараметрами на тестовом наборе
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Выводим отчёт по классификации
classification_rep = classification_report(y_test, y_pred)

# Визуализация
import matplotlib.pyplot as plt

history = best_model.fit(X_train_scaled, y_train, epochs=grid_search.best_params_['epochs'],
                         batch_size=grid_search.best_params_['batch_size'], validation_data=(X_test_scaled, y_test))

plt.figure(figsize=(12, 6))

# Потери
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Точность
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history)
