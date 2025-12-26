import pandas as pd
import yfinance as yf
import os
from datetime import datetime


class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_from_yahoo(self, start_date='2018-01-01', end_date='2023-12-31'):
        """Загрузка финансовых данных с Yahoo Finance"""

        # Тикеры для российского рынка и товаров
        tickers = {
            'Brent': 'BZ=F',  # Нефть Brent
            'USD_RUB': 'RUB=X',  # Курс USD/RUB
            'MOEX': 'IMOEX.ME',  # Индекс МосБиржи
            'Gold': 'GC=F',  # Золото
            'SBER': 'SBER.ME',  # Сбербанк
            'GAZP': 'GAZP.ME'  # Газпром
        }

        print("📥 Загрузка данных с Yahoo Finance...")
        data = yf.download(
            list(tickers.values()),
            start=start_date,
            end=end_date,
            progress=False
        )

        # Берем только цены закрытия
        close_prices = data['Close'].copy()
        close_prices.columns = list(tickers.keys())

        # Сохраняем сырые данные
        raw_path = os.path.join(self.data_dir, 'raw_dataset.csv')
        close_prices.to_csv(raw_path)
        print(f"✅ Данные сохранены в {raw_path}")

        return close_prices

    def load_from_csv(self, filename='raw_dataset.csv'):
        """Загрузка данных из CSV файла"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            raise FileNotFoundError(f"Файл {filepath} не найден")