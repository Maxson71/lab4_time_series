import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

from modules.data_loader import TimeSeriesLoader
from modules.decomposition import TimeSeriesDecomposition
from modules.properties import TimeSeriesProperties
from modules.sliding_window import ModelEvaluator
from modules.deep_learning import DeepLearningModel


def main():
    valcode = "EUR"
    window_size = 60
    future_steps = 30

    output_dir = Path(valcode) / "lab7"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ЛАБОРАТОРНА РОБОТА №7: METODI ГЛИБИННОГО НАВЧАННЯ (Deep Learning)")
    print("=" * 70)

    print("\n[1] Завантаження та дослідження даних...")
    loader = TimeSeriesLoader(valcode=valcode)
    df = loader.get_or_fetch_data()
    ts_data = df.set_index('date')['rate']

    decomp = TimeSeriesDecomposition(ts_data)
    stl_res = decomp.stl_decompose()
    decomp.plot_decomposition(output_dir / "decomposition_analysis.png")
    print("✓ Декомпозиція виконана та збережена")

    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]

    full_dataset_values = ts_data.values.reshape(-1, 1)
    train_values = full_dataset_values[:train_size]
    test_values = full_dataset_values[train_size - window_size:]

    print(f"✓ Розмір вибірок: Train={len(train_values)}, Test={len(test_values) - window_size}")

    # Побудова та навчання моделі
    print("\n[2] Конструювання нейронної мережі LSTM...")
    dl_model = DeepLearningModel(ts_data, window_size=window_size)

    # Створення архітектури
    model = dl_model.build_lstm_model(units=64, dropout_rate=0.2)

    print("\nАрхітектура моделі:")
    model.summary()
    dl_model.save_architecture_plot(output_dir / "model_architecture.png")
    print("\n[3] Навчання моделі...")
    history = dl_model.train(train_values, val_data=test_values, epochs=50, batch_size=32)

    # Графік Loss
    dl_model.plot_loss(output_dir / "training_loss.png")
    print("✓ Графік функції втрат збережено")

    # Валідація на тестовій вибірці
    print("\n[4] Валідація результатів (Test Set)...")

    # Прогноз на тестовій частині (всередині інтервалу спостереження)
    predictions = dl_model.predict(test_values)

    valid_series = pd.Series(test_data.values, index=test_data.index)
    pred_series = pd.Series(predictions.flatten(), index=test_data.index)

    # Оцінка метрик
    metrics = ModelEvaluator.calculate_metrics(valid_series.values, pred_series.values)
    print("\nМетрики якості моделі (Test Set):")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²:   {metrics['R2']:.4f}")

    # Візуалізація результатів
    plt.figure(figsize=(14, 7))
    plt.plot(train_data.index, train_data.values, label='Навчальні дані')
    plt.plot(valid_series.index, valid_series.values, label='Реальні дані (Test)', color='green')
    plt.plot(pred_series.index, pred_series.values, label='Прогноз LSTM', color='red', linestyle='--')
    plt.title('Результати роботи LSTM на тестовій вибірці')
    plt.xlabel('Дата')
    plt.ylabel('Курс')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "validation_results.png")
    plt.close()

    # 4. Прогнозування майбутнього (екстраполяція)
    print(f"\n[5] Прогнозування майбутнього на {future_steps} днів...")

    future_forecast = dl_model.forecast_future(steps=future_steps)

    # Створення дат для майбутнього
    last_date = ts_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

    future_series = pd.Series(future_forecast, index=future_dates)

    # Візуалізація з прогнозом
    plt.figure(figsize=(14, 7))
    # Покажемо останні 200 днів для кращої видимості
    zoom_data = ts_data[-200:]
    plt.plot(zoom_data.index, zoom_data.values, label='Історія (останні 200 днів)')
    plt.plot(future_series.index, future_series.values, label='Прогноз майбутнього', color='purple', linewidth=2)
    plt.title(f'Екстраполяція курсу {valcode} на {future_steps} днів')
    plt.xlabel('Дата')
    plt.ylabel('Курс')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "future_forecast.png")
    plt.close()

    print(f"\nПрогноз на наступні 5 днів:")
    print(future_series.head().to_string())

    # 5. Збереження звіту
    with open(output_dir / "lab7_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "=" * 70)
    print("РОБОТУ ЗАВЕРШЕНО")
    print(f"Результати збережено у: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()