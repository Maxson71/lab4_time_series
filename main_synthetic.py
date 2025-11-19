"""
Генерація та верифікація синтетичних часових рядів
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from modules.data_loader import TimeSeriesLoader
from modules.synthesis import SyntheticTimeSeriesGenerator
from modules.properties import TimeSeriesProperties
from modules.decomposition import TimeSeriesDecomposition


def main():
    valcode = "EUR"
    output_dir = Path(valcode) / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ГЕНЕРАЦІЯ СИНТЕТИЧНИХ ЧАСОВИХ РЯДІВ")
    print("=" * 60)

    # 1. Завантаження реальних даних та їх властивостей
    print("\nЗавантаження реальних даних...")
    loader = TimeSeriesLoader(valcode=valcode)
    real_data = loader.load_data()

    with open(Path(valcode) / "properties.json", 'r', encoding='utf-8') as f:
        properties = json.load(f)

    # Завантаження даних з трендом
    trend_data = pd.read_csv(Path(valcode) / "data_with_trend.csv")

    print(f"✓ Завантажено {len(real_data)} реальних записів")
    print(f"✓ Тип тренду: {trend_data['trend_kind'].iloc[0]}")

    # 2. Аналіз параметрів для синтезу
    print("\n" + "=" * 60)
    print("АНАЛІЗ ПАРАМЕТРІВ ДЛЯ ГЕНЕРАЦІЇ")
    print("=" * 60)

    # Параметри тренду
    y = real_data['rate'].values
    t = np.arange(len(y)).reshape(-1, 1)

    if trend_data['trend_kind'].iloc[0] == 'quadratic':
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression

        poly = PolynomialFeatures(degree=2, include_bias=True)
        T2 = poly.fit_transform(t)
        model = LinearRegression().fit(T2, y)
        trend_coeffs = [model.intercept_, model.coef_[1], model.coef_[2]]
        print(f"Квадратичний тренд: y = {trend_coeffs[0]:.4f} + {trend_coeffs[1]:.4f}*t + {trend_coeffs[2]:.6f}*t²")
    else:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(t, y)
        trend_coeffs = [model.intercept_, model.coef_[0]]
        print(f"Лінійний тренд: y = {trend_coeffs[0]:.4f} + {trend_coeffs[1]:.4f}*t")

    # Параметри залишків
    residuals = trend_data['residual'].values
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals, ddof=1)

    print(f"\nПараметри шуму:")
    print(f"  Середнє: {residual_mean:.4f}")
    print(f"  Станд. відхилення: {residual_std:.4f}")

    # 3. Генерація синтетичних даних
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦІЯ СИНТЕТИЧНИХ РЯДІВ")
    print("=" * 60)

    generator = SyntheticTimeSeriesGenerator(real_data, seed=42)

    # Модель 1: Тільки тренд + білий шум
    print("\n[1] Модель з трендом та білим шумом...")
    synthetic_properties = {
        'trend_type': trend_data['trend_kind'].iloc[0],
        'trend_coeffs': trend_coeffs,
        'noise_mean': residual_mean,
        'noise_std': residual_std,
        'skew': properties['statistics']['skew']
    }

    synth_1 = generator.generate_from_properties(synthetic_properties, n=len(real_data))
    synth_1.to_csv(output_dir / "synthetic_trend_noise.csv", index=False)
    print(f"✓ Згенеровано {len(synth_1)} точок")

    # Модель 2: ARMA процес
    print("\n[2] ARMA модель...")
    # Оцінка AR параметрів з автокореляції
    ar_param = properties['autocorrelation']['acf_lag1']
    synth_2 = generator.generate_arma(
        n=len(real_data),
        ar_params=[ar_param * 0.95],  # Злегка зменшено для стабільності
        ma_params=[0.1],
        noise_std=residual_std
    )
    # Додавання тренду
    synth_2['value'] = synth_2['value'] + trend_data['y_trend'].values
    synth_2.to_csv(output_dir / "synthetic_arma.csv", index=False)
    print(f"✓ Згенеровано {len(synth_2)} точок")

    # Модель 3: З врахуванням сезонності
    print("\n[3] Модель з трендом та сезонністю...")
    synth_3 = generator.generate_with_seasonality(
        n=len(real_data),
        seasonal_periods=[7, 30],  # Тижнева та місячна
        seasonal_amplitudes=[0.1, 0.15],
        noise_std=residual_std * 0.8
    )
    # Додавання тренду
    synth_3['value'] = synth_3['value'] + trend_data['y_trend'].values
    synth_3.to_csv(output_dir / "synthetic_seasonal.csv", index=False)
    print(f"✓ Згенеровано {len(synth_3)} точок")

    # 4. Верифікація адекватності
    print("\n" + "=" * 60)
    print("ВЕРИФІКАЦІЯ СИНТЕТИЧНИХ ДАНИХ")
    print("=" * 60)

    models = [
        ("Тренд + шум", synth_1),
        ("ARMA", synth_2),
        ("Тренд + сезонність", synth_3)
    ]

    verification_results = {}

    for model_name, synth_data in models:
        print(f"\n{model_name}:")

        # Статистична верифікація
        verification = generator.verify_similarity(synth_data, 'rate')
        verification_results[model_name] = verification

        print(f"  Різниця середніх: {verification['mean_difference']:.4f} "
              f"({verification['mean_relative_error'] * 100:.2f}%)")
        print(f"  Різниця ст.відх.: {verification['std_difference']:.4f} "
              f"({verification['std_relative_error'] * 100:.2f}%)")
        print(f"  KS-тест: p={verification['ks_pvalue']:.4f}, "
              f"{'✓ схожі' if verification['ks_similar'] else '✗ відрізняються'}")
        print(f"  T-тест: p={verification['t_pvalue']:.4f}, "
              f"{'✓ схожі' if verification['t_similar'] else '✗ відрізняються'}")

        # Порівняння властивостей
        synth_props = TimeSeriesProperties(synth_data['value'])
        synth_stats = synth_props.compute_statistics()

        print(f"  Властивості:")
        print(f"    Mean: real={properties['statistics']['mean']:.4f}, "
              f"synth={synth_stats['mean']:.4f}")
        print(f"    Std:  real={properties['statistics']['std']:.4f}, "
              f"synth={synth_stats['std']:.4f}")
        print(f"    Skew: real={properties['statistics']['skew']:.4f}, "
              f"synth={synth_stats['skew']:.4f}")

    # Збереження результатів верифікації з явною конвертацією типів
    verification_output = {}
    for model_name, result in verification_results.items():
        verification_output[model_name] = {
            'mean_difference': float(result['mean_difference']),
            'std_difference': float(result['std_difference']),
            'mean_relative_error': float(result['mean_relative_error']),
            'std_relative_error': float(result['std_relative_error']),
            'ks_statistic': float(result['ks_statistic']),
            'ks_pvalue': float(result['ks_pvalue']),
            'ks_similar': bool(result['ks_similar']),
            't_statistic': float(result['t_statistic']),
            't_pvalue': float(result['t_pvalue']),
            't_similar': bool(result['t_similar'])
        }

    with open(output_dir / "verification_results.json", 'w', encoding='utf-8') as f:
        json.dump(verification_output, f, indent=2, ensure_ascii=False)

    print("✓ Результати верифікації збережено")

    # 5. Візуалізація порівняння
    print("\n" + "=" * 60)
    print("ВІЗУАЛІЗАЦІЯ ПОРІВНЯННЯ")
    print("=" * 60)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Реальні дані
    axes[0].plot(real_data['rate'].values, label='Реальні дані', color='black', linewidth=1.5)
    axes[0].set_title('Реальні дані EUR/UAH', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Курс (грн)', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Синтетичні моделі
    for idx, (model_name, synth_data) in enumerate(models, start=1):
        axes[idx].plot(synth_data['value'].values, label=f'Синтетична: {model_name}',
                       color='blue', linewidth=1.5, alpha=0.7)
        axes[idx].plot(real_data['rate'].values, label='Реальні (для порівняння)',
                       color='black', linewidth=0.8, alpha=0.3)
        axes[idx].set_title(f'Модель: {model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Курс (грн)', fontsize=10)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Час (дні)', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_models.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Візуалізація збережена")

    # Порівняння розподілів
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Гістограми
    axes[0, 0].hist(real_data['rate'], bins=30, alpha=0.7, label='Реальні', color='black', density=True)
    axes[0, 0].set_title('Розподіл: Реальні дані')
    axes[0, 0].set_xlabel('Курс (грн)')
    axes[0, 0].set_ylabel('Щільність')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for idx, (model_name, synth_data) in enumerate(models):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        axes[row, col].hist(synth_data['value'], bins=30, alpha=0.7,
                            label=model_name, color='blue', density=True)
        axes[row, col].hist(real_data['rate'], bins=30, alpha=0.3,
                            label='Реальні', color='black', density=True)
        axes[row, col].set_title(f'Розподіл: {model_name}')
        axes[row, col].set_xlabel('Курс (грн)')
        axes[row, col].set_ylabel('Щільність')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Порівняння розподілів збережено")

    # QQ-plots
    from scipy import stats as sp_stats

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (model_name, synth_data) in enumerate(models):
        sp_stats.probplot(synth_data['value'], dist="norm", plot=axes[idx])
        axes[idx].set_title(f'Q-Q Plot: {model_name}')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "qq_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Q-Q plots збережено")

    print("\n" + "=" * 60)
    print("ГЕНЕРАЦІЯ ТА ВЕРИФІКАЦІЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"\nРезультати збережено у папці: {output_dir}/")

    # Підсумкова таблиця
    print("\nПідсумкова таблиця верифікації:")
    print("-" * 80)
    print(f"{'Модель':<25} {'Δ Mean':<12} {'Δ Std':<12} {'KS p-val':<12} {'Схожість':<12}")
    print("-" * 80)
    for model_name, result in verification_results.items():
        similarity = "✓ Так" if result['ks_similar'] and result['t_similar'] else "✗ Ні"
        print(f"{model_name:<25} {result['mean_difference']:<12.4f} "
              f"{result['std_difference']:<12.4f} {result['ks_pvalue']:<12.4f} {similarity:<12}")
    print("-" * 80)


if __name__ == "__main__":
    main()
