"""
Модуль кластеризації часових рядів
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import euclidean
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    print("Warning: tslearn not available. Using basic clustering only.")


class TimeSeriesClustering:
    """Клас для кластеризації часових рядів"""

    def __init__(self, data: pd.DataFrame):
        """
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame з колонками: date та різні часові ряди
        """
        self.data = data
        self.clusters = None
        self.labels = None

    def _prepare_window_features(self, series: pd.Series, window: int = 30) -> np.ndarray:
        """
        Створення ковзного вікна для кластеризації

        Returns:
        --------
        np.ndarray : матриця (n_windows, window_size)
        """
        values = series.values
        n = len(values)
        n_windows = n - window + 1

        windows = np.array([values[i:i + window] for i in range(n_windows)])
        return windows

    def classical_kmeans(self, n_clusters: int = 3, window: int = 30) -> Dict:
        """
        Класична кластеризація K-means з евклідовою метрикою

        Parameters:
        -----------
        n_clusters : int
            Кількість кластерів
        window : int
            Розмір вікна для сегментації
        """
        # Припустимо, що у нас одна колонка з даними (rate)
        if 'rate' not in self.data.columns:
            raise ValueError("Column 'rate' not found in data")

        windows = self._prepare_window_features(self.data['rate'], window)

        # Нормалізація
        scaler = StandardScaler()
        windows_scaled = scaler.fit_transform(windows)

        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = kmeans.fit_predict(windows_scaled)

        return {
            'labels': self.labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'n_clusters': n_clusters
        }

    def dtw_kmeans(self, n_clusters: int = 3) -> Dict:
        """
        Кластеризація з використанням DTW (Dynamic Time Warping)
        Потребує бібліотеку tslearn
        """
        if not TSLEARN_AVAILABLE:
            raise ImportError("tslearn is required for DTW clustering. Install it with: pip install tslearn")

        # Підготовка даних у форматі tslearn
        series = self.data['rate'].values.reshape(-1, 1)

        # Нормалізація
        scaler = TimeSeriesScalerMeanVariance()
        series_scaled = scaler.fit_transform(series.reshape(1, -1, 1))

        # Розділення на вікна для кластеризації
        window = 30
        windows = self._prepare_window_features(self.data['rate'], window)
        windows = windows.reshape(windows.shape[0], windows.shape[1], 1)

        # DTW K-means
        model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw",
            max_iter=10,
            random_state=42
        )
        self.labels = model.fit_predict(windows)

        return {
            'labels': self.labels,
            'centers': model.cluster_centers_,
            'inertia': model.inertia_,
            'n_clusters': n_clusters
        }

    def hierarchical_clustering(self, n_clusters: int = 3, method: str = 'ward') -> Dict:
        """
        Ієрархічна кластеризація

        Parameters:
        -----------
        n_clusters : int
            Кількість кластерів
        method : str
            Метод зв'язування: 'ward', 'complete', 'average', 'single'
        """
        windows = self._prepare_window_features(self.data['rate'], window=30)

        # Нормалізація
        scaler = StandardScaler()
        windows_scaled = scaler.fit_transform(windows)

        # Ієрархічна кластеризація
        linkage_matrix = linkage(windows_scaled, method=method)
        self.labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        return {
            'labels': self.labels,
            'linkage_matrix': linkage_matrix,
            'n_clusters': n_clusters
        }

    def plot_clusters(self, save_path: str = None):
        """Візуалізація результатів кластеризації"""
        if self.labels is None:
            raise ValueError("Спочатку виконайте кластеризацію!")

        unique_labels = np.unique(self.labels)
        n_clusters = len(unique_labels)

        fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3 * n_clusters))
        if n_clusters == 1:
            axes = [axes]

        window = 30
        windows = self._prepare_window_features(self.data['rate'], window)

        for i, label in enumerate(unique_labels):
            cluster_windows = windows[self.labels == label]

            axes[i].plot(cluster_windows.T, alpha=0.3, color='blue')
            axes[i].plot(cluster_windows.mean(axis=0), color='red', linewidth=2, label='Центроїд')
            axes[i].set_title(f'Кластер {label} (n={len(cluster_windows)})')
            axes[i].set_xlabel('Час')
            axes[i].set_ylabel('Значення')
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

    def multifactor_clustering(self, factors: List[str], n_clusters: int = 3) -> Dict:
        """
        Багатофакторна кластеризація

        Parameters:
        -----------
        factors : List[str]
            Список колонок для кластеризації
        n_clusters : int
            Кількість кластерів
        """
        # Створення матриці ознак
        feature_matrix = self.data[factors].values

        # Нормалізація
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)

        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = kmeans.fit_predict(features_scaled)

        return {
            'labels': self.labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'factors': factors
        }
