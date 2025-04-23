import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# === 1. Загрузка и подготовка данных ===
df = pd.read_csv("synthetic_behavior_risk_balanced.csv")

# Удаляем нечисловые и текстовые признаки
X = df.drop(columns=['user_id', 'session_id', 'timestamp_start', 'actions_seq', 'risk_level'])
y = df['risk_level'] - 1  # уровни риска от 0 до 4

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на train/test (будем обучать только на risk_level == 0)
X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
X_train = X_train_full[y_train_full == 0]  # автоэнкодер видит только "безопасные" данные

print("Risk level distribution in training set:")
print(y_train_full.value_counts().sort_index())
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
# === 2. Архитектура автоэнкодера ===
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# === 3. Инициализация ===
input_dim = X_train.shape[1]
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 4. Обучение ===
epochs = 30
batch_size = 64
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch = X_train_tensor[indices]
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} completed.")

# === 5. Оценка ===
model.eval()
with torch.no_grad():
    reconstructed = model(X_test_tensor)
    reconstruction_error = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).numpy()

# Квантильные пороги ошибок на основе нормального поведения
thresholds = np.percentile(reconstruction_error, [20, 40, 60, 80])
predicted_risk = np.digitize(reconstruction_error, bins=thresholds)

# === 6. Метрики ===
print("\n=== Autoencoder Risk Level Prediction Report ===")
print(classification_report(y_test_tensor, predicted_risk, digits=4))
