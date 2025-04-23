import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report




# === 1. Загрузка и подготовка данных ===
df = pd.read_csv("synthetic_behavior_risk_balanced.csv")  # путь к новому датасету

# Удаляем нечисловые и текстовые признаки
X = df.drop(columns=['user_id', 'session_id', 'timestamp_start', 'actions_seq', 'risk_level'])
y = df['risk_level'] - 1  # приводим метки к диапазону 0–4 для CrossEntropyLoss

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Деление на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Преобразуем в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)


print("Risk level distribution in training set:")
print(y_train.value_counts().sort_index())

# === 2. Определение модели MLP ===
class RiskMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(RiskMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# === 3. Инициализация ===
input_dim = X_train.shape[1]
num_classes = 5  # классы от 0 до 4 → уровни риска 1–5
model = RiskMLP(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 4. Обучение ===
epochs = 20
batch_size = 64
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} completed.")

# === 5. Оценка ===
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted_classes = torch.max(outputs, 1)
    report = classification_report(y_test_tensor, predicted_classes, digits=4)
    print("\n=== Risk Level Classification Report ===")
    print(report)
