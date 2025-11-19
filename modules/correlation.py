"""
Модуль кореляційного аналізу часових рядів
"""
import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class TimeSeriesCorrelation:
    """Клас для кореляційного аналізу часових рядів"""

    def __init__(self, data: pd.DataFrame):
        """
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame з кількома часовими рядами
        """
        self.data = data

    def pearson_correlation(self, col1: str, col2: str) -> Dict:
        """
        Кореляція Пірсона між двома рядами

        Returns:
        --------
        Dict з коефіцієнтом кореляції та p-value
        """
        corr, pvalue = stats.pearsonr(
            self.data[col1].dropna(),
            self.data[col2].dropna()
        )

        return {
            'correlation': float(corr),
            'pvalue': float(pvalue),
            'is_significant': pvalue < 0.05
        }

    def cross_correlation(self, col1: str, col2: str, maxlag: int = 50) -> Dict:
        """
        Взаємна кореляція (cross-correlation) з часовим зсувом

        Parameters:
        -----------
        col1, col2 : str
            Назви колонок для аналізу
        maxlag : int
            Максимальний часовий зсув

        Returns:
        --------
        Dict з масивом кореляцій та оптимальним лагом
        """
        series1 = self.data[col1].dropna().values
        series2 = self.data[col2].dropna().values

        # Нормалізація
        series1 = (series1 - np.mean(series1)) / np.std(series1)
        series2 = (series2 - np.mean(series2)) / np.std(series2)

        # Взаємна кореляція
        correlation = signal.correlate(series1, series2, mode='full', method='auto')
        lags = signal.correlation_lags(len(series1), len(series2), mode='full')

        # Обмеження до maxlag
        mask = np.abs(lags) <= maxlag
        correlation = correlation[mask]
        lags = lags[mask]

        # Нормалізація
        correlation = correlation / len(series1)

        # Знаходження максимальної кореляції
        max_corr_idx = np.argmax(np.abs(correlation))
        optimal_lag = lags[max_corr_idx]
        max_correlation = correlation[max_corr_idx]

        return {
            'correlations': correlation,
            'lags': lags,
            'optimal_lag': int(optimal_lag),
            'max_correlation': float(max_correlation)
        }

    def lagged_correlation(self, col: str, max_lag: int = 30) -> Dict:
        """
        Автокореляція з різними лагами

        Parameters:
        -----------
        col : str
            Назва колонки
        max_lag : int
            Максимальний лаг
        """
        series = self.data[col].dropna()
        correlations = []

        for lag in range(max_lag + 1):
            if lag == 0:
                corr = 1.0
            else:
                corr = series.autocorr(lag=lag)
            correlations.append(corr)

        return {
            'lags': list(range(max_lag + 1)),
            'correlations': correlations
        }

    def correlation_matrix(self, columns: List[str] = None) -> pd.DataFrame:
        """
        Матриця кореляцій між кількома рядами

        Parameters:
        -----------
        columns : List[str]
            Список колонок для аналізу (якщо None - всі числові)
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = self.data[columns].corr(method='pearson')
        return corr_matrix

    def granger_causality(self, col1: str, col2: str, maxlag: int = 10) -> Dict:
        """
        Тест причинності Грейнджера (потребує statsmodels)

        Перевіряє чи допомагає col1 передбачати col2
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        data = self.data[[col2, col1]].dropna()

        try:
            results = grangercausalitytests(data, maxlag=maxlag, verbose=False)

            # Збір p-values для різних лагів
            pvalues = {}
            for lag in range(1, maxlag + 1):
                # F-test p-value
                pvalues[lag] = results[lag][0]['ssr_ftest'][1]

            # Визначення чи є значущою причинність на будь-якому лагу
            is_causal = any(p < 0.05 for p in pvalues.values())

            return {
                'pvalues': pvalues,
                'is_causal': is_causal,
                'best_lag': min(pvalues, key=pvalues.get)
            }
        except Exception as e:
            return {'error': str(e)}

    def plot_correlation_matrix(self, columns: List[str] = None, save_path: str = None):
        """Візуалізація матриці кореляцій"""
        corr_matrix = self.correlation_matrix(columns)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Матриця кореляцій часових рядів')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_cross_correlation(self, col1: str, col2: str, maxlag: int = 50, save_path: str = None):
        """Візуалізація взаємної кореляції"""
        result = self.cross_correlation(col1, col2, maxlag)

        plt.figure(figsize=(12, 5))
        plt.stem(result['lags'], result['correlations'], basefmt=' ')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Лаг')
        plt.ylabel('Кореляція')
        plt.title(f'Взаємна кореляція: {col1} vs {col2}\n'
                  f'Оптимальний лаг: {result["optimal_lag"]}, '
                  f'Макс. кореляція: {result["max_correlation"]:.3f}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()
