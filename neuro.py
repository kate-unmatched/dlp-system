import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


# === 1. Загрузка датасета ===
df = pd.read_csv("synthetic_behavior_dataset_complex.csv")  # путь к твоему CSV-файлу

# === 2. Предобработка ===
# Удалим нечисловые и неиспользуемые признаки
X = df.drop(columns=['user_id', 'session_id', 'timestamp_start', 'actions_seq', 'is_anomaly'])
y = df['is_anomaly']

# Масштабируем числовые признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CNN требует входа в форме (samples, channels, features)
X_scaled = np.expand_dims(X_scaled, axis=1)

# === 3. Разделение на обучающую и тестовую выборки ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Преобразование в PyTorch тензоры
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# === 4. Определим CNN-модель ===
class BehaviorCNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear((input_size // 2) * 32, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 класса: 0 — норма, 1 — аномалия

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(x)
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)

# === 5. Инициализация ===
input_size = X_train.shape[2]
model = BehaviorCNN(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 6. Обучение ===
epochs = 10
batch_size = 64
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} completed.")

# === 7. Оценка ===
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    _, predicted_classes = torch.max(predictions, 1)
    report = classification_report(y_test, predicted_classes, digits=4)
    print("\n=== Classification Report ===")
    print(report)
