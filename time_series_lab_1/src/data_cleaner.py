import pandas as pd
import numpy as np
import os


class DataCleaner:
    def __init__(self):
        self.cleaning_report = {}

    def clean_data(self, df):
        """Полная очистка временного ряда"""

        print("🧹 Начало очистки данных...")
        original_shape = df.shape

        # 1. Проверка индекса
        df = self._ensure_datetime_index(df)

        # 2. Удаление дубликатов
        df = self._remove_duplicates(df)

        # 3. Обработка пропусков
        df = self._handle_missing_values(df)

        # 4. Обработка выбросов
        df_cleaned = self._handle_outliers(df)

        # 5. Сохранение очищенных данных
        cleaned_path = os.path.join('data', 'cleaned_dataset.csv')
        df_cleaned.to_csv(cleaned_path)

        print(f"✅ Очистка завершена: {original_shape} -> {df_cleaned.shape}")

        return df_cleaned

    def _ensure_datetime_index(self, df):
        """Приведение индекса к datetime"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    def _remove_duplicates(self, df):
        """Удаление дубликатов по времени"""
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            print(f"🔄 Удалено дубликатов: {duplicates}")
            df = df[~df.index.duplicated(keep='first')]
        return df

    def _handle_missing_values(self, df):
        """Обработка пропущенных значений"""
        missing_before = df.isnull().sum().sum()

        if missing_before > 0:
            print(f"🔍 Найдено пропусков: {missing_before}")

            # Интерполяция для временных рядов
            df_filled = df.interpolate(method='time', limit_direction='both')

            # Если остались пропуски - forward fill
            df_filled = df_filled.ffill().bfill()

            missing_after = df_filled.isnull().sum().sum()
            print(f"✅ Пропуски обработаны: осталось {missing_after}")

            return df_filled
        return df

    def _handle_outliers(self, df, method='iqr'):
        """Обработка выбросов методом IQR"""
        df_clean = df.copy()

        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Находим выбросы
            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()

            if outliers > 0:
                print(f"📊 {column}: найдено {outliers} выбросов")
                # Заменяем выбросы на границы
                df_clean[column] = np.where(df_clean[column] < lower_bound, lower_bound, df_clean[column])
                df_clean[column] = np.where(df_clean[column] > upper_bound, upper_bound, df_clean[column])

        return df_clean

    def get_cleaning_report(self):
        """Отчет по очистке данных"""
        return self.cleaning_report