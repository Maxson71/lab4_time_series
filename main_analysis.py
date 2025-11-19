"""
Головний скрипт для поглибленого аналізу реальних даних Time Series
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

from modules.data_loader import TimeSeriesLoader
from modules.decomposition import TimeSeriesDecomposition
from modules.properties import TimeSeriesProperties
from modules.clustering import TimeSeriesClustering
from modules.correlation import TimeSeriesCorrelation

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


def main():
    # Конфігурація
    valcode = "EUR"
    output_dir = Path(valcode)
    output_dir.mkdir(exist_ok=True)

    # 1. Завантаження даних
    print("=" * 60)
    print("ЕТАП 1: Завантаження даних")
    print("=" * 60)

    loader = TimeSeriesLoader(valcode=valcode)
    df = loader.get_or_fetch_data()
    print(f"Завантажено {len(df)} записів")
    print(df.head())

    # 2. Декомпозиція
    print("\n" + "=" * 60)
    print("ЕТАП 2: Декомпозиція часового ряду")
    print("=" * 60)

    decomposer = TimeSeriesDecomposition(df.set_index('date')['rate'], freq=30)

    # Класична декомпозиція
    classic_result = decomposer.classical_decompose(model='additive')
    decomposer.plot_decomposition(output_dir / "decomposition_classical.png")
    print("✓ Класична декомпозиція виконана")

    # STL декомпозиція
    stl_result = decomposer.stl_decompose(seasonal=7)
    decomposer.plot_decomposition(output_dir / "decomposition_stl.png")
    print("✓ STL декомпозиція виконана")

    # Збереження результатів декомпозиції
    decomp_df = pd.DataFrame({
        'date': df['date'],
        'original': df['rate'],
        'trend': stl_result['trend'],
        'seasonal': stl_result['seasonal'],
        'residual': stl_result['residual']
    })
    decomp_df.to_csv(output_dir / "decomposition_results.csv", index=False)

    # 3. Виявлення властивостей
    print("\n" + "=" * 60)
    print("ЕТАП 3: Виявлення властивостей часового ряду")
    print("=" * 60)

    props = TimeSeriesProperties(df['rate'])

    # Базові статистики
    stats_results = props.compute_statistics()
    print("\nБазові статистики:")
    for key, value in stats_results.items():
        print(f"  {key}: {value:.4f}")

    # Тести стаціонарності
    stationarity = props.test_stationarity()
    print("\nСтаціонарність:")
    print(f"  ADF тест: p-value = {stationarity['adf_pvalue']:.4f}, "
          f"{'стаціонарний' if stationarity['adf_is_stationary'] else 'нестаціонарний'}")
    print(f"  KPSS тест: p-value = {stationarity['kpss_pvalue']:.4f}, "
          f"{'стаціонарний' if stationarity['kpss_is_stationary'] else 'нестаціонарний'}")

    # Автокореляція
    autocorr = props.compute_autocorrelation(nlags=40)
    print(f"\nАвтокореляція (lag=1): {autocorr['acf_lag1']:.4f}")

    # Викиди
    outliers = props.detect_outliers(method='iqr')
    print(f"\nВикиди: {outliers['outlier_count']} ({outliers['outlier_percentage']:.2f}%)")

    # Збереження властивостей з конвертацією типів
    properties_output = {
        'statistics': {k: float(v) for k, v in stats_results.items()},
        'stationarity': {
            'adf_statistic': float(stationarity['adf_statistic']),
            'adf_pvalue': float(stationarity['adf_pvalue']),
            'adf_is_stationary': bool(stationarity['adf_is_stationary']),
            'kpss_statistic': float(stationarity['kpss_statistic']),
            'kpss_pvalue': float(stationarity['kpss_pvalue']),
            'kpss_is_stationary': bool(stationarity['kpss_is_stationary'])
        },
        'autocorrelation': {
            'acf_lag1': float(autocorr['acf_lag1'])
        },
        'outliers': {
            'outlier_count': int(outliers['outlier_count']),
            'outlier_percentage': float(outliers['outlier_percentage']),
            'outlier_indices': [int(x) for x in outliers['outlier_indices']]
        }
    }

    with open(output_dir / "properties.json", 'w', encoding='utf-8') as f:
        json.dump(properties_output, f, indent=2, ensure_ascii=False)

    print("✓ Властивості збережено у properties.json")

    # 4. Аналіз тренду (з попередньої роботи)
    print("\n" + "=" * 60)
    print("ЕТАП 4: Аналіз тренду")
    print("=" * 60)

    y = df["rate"].astype(float).values
    t = np.arange(len(y)).reshape(-1, 1)

    # Лінійний тренд
    lin = LinearRegression().fit(t, y)
    y_lin = lin.predict(t)
    r2_lin = r2_score(y, y_lin)

    # Квадратичний тренд
    poly = PolynomialFeatures(degree=2, include_bias=True)
    T2 = poly.fit_transform(t)
    quad = LinearRegression().fit(T2, y)
    y_quad = quad.predict(T2)
    r2_quad = r2_score(y, y_quad)

    trend_kind = "quadratic" if r2_quad >= r2_lin else "linear"
    y_trend = y_quad if trend_kind == "quadratic" else y_lin

    print(f"Тип тренду: {trend_kind}")
    print(f"  R² (лінійний): {r2_lin:.4f}")
    print(f"  R² (квадратичний): {r2_quad:.4f}")

    # Створення DataFrame з трендом
    df_trend = df.copy()
    df_trend["trend_kind"] = trend_kind
    df_trend["y_trend"] = y_trend
    df_trend["residual"] = df_trend["rate"] - df_trend["y_trend"]
    df_trend.to_csv(output_dir / "data_with_trend.csv", index=False)

    # 5. Кластеризація
    print("\n" + "=" * 60)
    print("ЕТАП 5: Кластеризація часових рядів")
    print("=" * 60)

    clusterer = TimeSeriesClustering(df)

    # Класична кластеризація
    kmeans_result = clusterer.classical_kmeans(n_clusters=3, window=30)
    print(f"✓ K-means кластеризація виконана: {kmeans_result['n_clusters']} кластерів")
    print(f"  Inertia: {kmeans_result['inertia']:.2f}")
    clusterer.plot_clusters(output_dir / "clusters_kmeans.png")

    # Ієрархічна кластеризація
    hier_result = clusterer.hierarchical_clustering(n_clusters=3)
    print(f"✓ Ієрархічна кластеризація виконана: {hier_result['n_clusters']} кластерів")
    clusterer.plot_clusters(output_dir / "clusters_hierarchical.png")

    # 6. Кореляційний аналіз
    print("\n" + "=" * 60)
    print("ЕТАП 6: Кореляційний аналіз")
    print("=" * 60)

    # Створення додаткових рядів для аналізу
    df_corr = df.copy()
    df_corr['returns'] = df_corr['rate'].pct_change()
    df_corr['log_returns'] = np.log(df_corr['rate']).diff()
    df_corr['ma7'] = df_corr['rate'].rolling(window=7).mean()
    df_corr['ma30'] = df_corr['rate'].rolling(window=30).mean()

    correlator = TimeSeriesCorrelation(df_corr)

    # Матриця кореляцій
    corr_matrix = correlator.correlation_matrix(['rate', 'returns', 'log_returns', 'ma7', 'ma30'])
    print("\nМатриця кореляцій:")
    print(corr_matrix)
    correlator.plot_correlation_matrix(['rate', 'returns', 'log_returns', 'ma7', 'ma30'],
                                       output_dir / "correlation_matrix.png")

    # Взаємна кореляція
    if 'ma7' in df_corr.columns and df_corr['ma7'].notna().sum() > 50:
        cross_corr = correlator.cross_correlation('rate', 'ma7', maxlag=30)
        print(f"\nВзаємна кореляція (rate vs ma7):")
        print(f"  Оптимальний лаг: {cross_corr['optimal_lag']}")
        print(f"  Максимальна кореляція: {cross_corr['max_correlation']:.4f}")
        correlator.plot_cross_correlation('rate', 'ma7', maxlag=30,
                                          save_path=output_dir / "cross_correlation.png")

    print("\n" + "=" * 60)
    print("АНАЛІЗ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"\nРезультати збережено у папці: {output_dir}/")


if __name__ == "__main__":
    main()
