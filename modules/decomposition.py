"""
Модуль декомпозиції часових рядів
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from typing import Dict, Tuple
import matplotlib.pyplot as plt


class TimeSeriesDecomposition:
    """Клас для декомпозиції часових рядів"""

    def __init__(self, data: pd.Series, freq: int = None):
        """
        Parameters:
        -----------
        data : pd.Series
            Часовий ряд для декомпозиції
        freq : int
            Частота сезонності (наприклад, 365 для річної сезонності)
        """
        self.data = data
        self.freq = freq or self._estimate_frequency()
        self.result = None

    def _estimate_frequency(self) -> int:
        """Автоматична оцінка частоти сезонності"""
        # Для валютних даних часто є тижнева (7) або місячна (30) сезонність
        n = len(self.data)
        if n > 730:  # Більше 2 років
            return 365  # Річна сезонність
        elif n > 60:
            return 30  # Місячна сезонність
        else:
            return 7  # Тижнева сезонність

    def classical_decompose(self, model: str = 'additive') -> Dict:
        """
        Класична декомпозиція (moving averages)

        Parameters:
        -----------
        model : str
            'additive' або 'multiplicative'
        """
        self.result = seasonal_decompose(
            self.data,
            model=model,
            period=self.freq,
            extrapolate_trend='freq'
        )

        return {
            'trend': self.result.trend,
            'seasonal': self.result.seasonal,
            'residual': self.result.resid,
            'observed': self.result.observed
        }

    def stl_decompose(self, seasonal: int = 7, trend: int = None) -> Dict:
        """
        STL декомпозиція (Seasonal-Trend decomposition using LOESS)

        Parameters:
        -----------
        seasonal : int
            Довжина вікна для сезонності (повинна бути непарною)
        trend : int
            Довжина вікна для тренду
        """
        if seasonal % 2 == 0:
            seasonal += 1  # Має бути непарним

        stl = STL(self.data, seasonal=seasonal, trend=trend, period=self.freq)
        self.result = stl.fit()

        return {
            'trend': self.result.trend,
            'seasonal': self.result.seasonal,
            'residual': self.result.resid,
            'observed': self.data
        }

    def plot_decomposition(self, save_path: str = None):
        """Візуалізація результатів декомпозиції"""
        if self.result is None:
            raise ValueError("Спочатку виконайте декомпозицію!")

        fig = self.result.plot()
        fig.set_size_inches(12, 8)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()
