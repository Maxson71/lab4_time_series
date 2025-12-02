"""
Модуль глибокого навчання для часових рядів (Lab 7)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as pd_tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any

class DeepLearningModel:
    def __init__(self, data: pd.Series, window_size: int = 60):
        """
        Parameters:
        -----------
        data : pd.Series
            Вхідний часовий ряд
        window_size : int
            Розмір вікна (скільки попередніх днів використовуємо для прогнозу)
        """
        self.raw_data = data.values.reshape(-1, 1)
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None

        self.scaled_data = self.scaler.fit_transform(self.raw_data)

    def create_dataset(self, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Перетворення масиву в формат [samples, time_steps, features]
        """
        X, y = [], []
        for i in range(len(dataset) - self.window_size):
            X.append(dataset[i:(i + self.window_size), 0])
            y.append(dataset[i + self.window_size, 0])
        return np.array(X), np.array(y)

    def build_lstm_model(self, units: int = 50, dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Побудова архітектури LSTM
        """
        model = Sequential()
        model.add(Input(shape=(self.window_size, 1)))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model

    def train(self, train_data: np.ndarray, val_data: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32):
        """
        Навчання моделі
        """
        train_scaled = self.scaler.transform(train_data)

        X_train, y_train = self.create_dataset(train_scaled)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        validation_data = None
        if val_data is not None:
            val_scaled = self.scaler.transform(val_data)
            X_val, y_val = self.create_dataset(val_scaled)
            X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
            validation_data = (X_val, y_val)

        # Callback для зупинки, якщо навчання не покращується
        early_stop = EarlyStopping(monitor='val_loss' if val_data is not None else 'loss',
                                   patience=10, restore_best_weights=True)

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stop],
            verbose=1
        )
        return self.history

    def predict(self, data_slice: np.ndarray) -> np.ndarray:
        """
        Прогнозування на основі вхідних даних
        """
        data_scaled = self.scaler.transform(data_slice)

        X, _ = self.create_dataset(data_scaled)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)

    def forecast_future(self, steps: int) -> pd.DataFrame:
        """
        Рекурсивне прогнозування майбутнього (екстраполяція)
        """
        curr_input = self.scaled_data[-self.window_size:].reshape(1, self.window_size, 1)
        forecast = []

        for _ in range(steps):
            pred = self.model.predict(curr_input, verbose=0)
            forecast.append(pred[0, 0])
            curr_input = np.append(curr_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        forecast_array = np.array(forecast).reshape(-1, 1)
        forecast_real = self.scaler.inverse_transform(forecast_array)

        return forecast_real.flatten()

    def plot_loss(self, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Графік функції втрат (Model Loss)')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def save_architecture_plot(self, save_path):

        try:
            plot_model(self.model, to_file=save_path, show_shapes=True, show_layer_names=True)
        except Exception:
            print("Не вдалося зберегти графік архітектури.")