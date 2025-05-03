import numpy as np
import pandas as pd

# Функция для генерации синтетических данных с зависимыми метками
def generate_synthetic_data_with_labels(num_samples=5000):
    data = []
    for i in range(num_samples):
        # Генерация случайных значений признаков
        file_create_count = np.random.poisson(2)
        file_update_count = np.random.poisson(1)
        file_delete_count = np.random.poisson(0.5)
        file_access_sensitive_docs = np.random.poisson(1)
        file_sensitive_word_matches = np.random.poisson(2)
        file_contains_card_number = np.random.binomial(1, 0.05)
        file_contains_passport_data = np.random.binomial(1, 0.05)
        file_confidentiality_score = np.random.uniform(0, 1)
        archive_created_count = np.random.poisson(1)
        file_permission_changed_count = np.random.poisson(0.2)
        file_system_update_count = np.random.poisson(0.3)

        # Логика для вычисления danger_score
        danger_score = 0

        # Пример логики для присваивания меток
        if file_create_count > 5 or file_update_count > 5:
            danger_score += 2  # Высокая активность в файловой системе увеличивает опасность
        if file_access_sensitive_docs > 2:
            danger_score += 2  # Доступ к чувствительным файлам
        if file_sensitive_word_matches > 5:
            danger_score += 2  # Частые вхождения чувствительных слов
        if file_contains_card_number or file_contains_passport_data:
            danger_score += 3  # Чувствительная информация в файлах

        # Сетевая активность
        if file_permission_changed_count > 3:
            danger_score += 2  # Изменение прав доступа
        if archive_created_count > 3:
            danger_score += 1  # Создание архива может указывать на попытку скрыть данные

        # Применение пороговых значений для присваивания меток
        # danger_score будет от 0 до 7, но можно адаптировать для нужных значений.
        if danger_score > 6:
            danger_score = 6
        else:
            danger_score = int(danger_score)

        # Добавляем признаки и метку в список
        data.append([
            file_create_count, file_update_count, file_delete_count,
            file_access_sensitive_docs, file_sensitive_word_matches,
            file_contains_card_number, file_contains_passport_data, file_confidentiality_score,
            archive_created_count, file_permission_changed_count, file_system_update_count,
            danger_score
        ])

    # Создаём DataFrame
    columns = [
        'file_create_count', 'file_update_count', 'file_delete_count', 'file_access_sensitive_docs',
        'file_sensitive_word_matches', 'file_contains_card_number', 'file_contains_passport_data',
        'file_confidentiality_score', 'archive_created_count', 'file_permission_changed_count',
        'file_system_update_count', 'danger_score'
    ]
    df = pd.DataFrame(data, columns=columns)
    return df


# Генерация данных
df = generate_synthetic_data_with_labels()

# Проверка сгенерированных данных
df.head()
