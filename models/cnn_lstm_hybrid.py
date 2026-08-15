#!/usr/bin/env python3
"""
cnn_lstm_hybrid.py
==================
CNN-LSTM Hybrid model for spatial-temporal climate feature extraction.

Architecture:
  Input → Conv1D (64, kernel=3) → BatchNorm → Conv1D (128, kernel=3) →
  MaxPool1D → LSTM (64) → Attention Pooling → Dense → Output

Combines spatial filters (CNN) with temporal memory (LSTM).
"""

import numpy as np
import json
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


class AttentionPooling(layers.Layer):
    """Attention-based pooling over LSTM sequence outputs."""
    
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.attention_dense = layers.Dense(units, activation="tanh")
        self.context_vector = layers.Dense(1, use_bias=False)
    
    def call(self, x):
        # x shape: (batch, seq_len, features)
        attention_scores = self.context_vector(self.attention_dense(x))
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        context = tf.reduce_sum(x * attention_weights, axis=1)
        return context
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_cnn_lstm_model(
    input_shape,
    conv_filters=[64, 128],
    kernel_size=3,
    lstm_units=64,
    dense_units=32,
    dropout_rate=0.3,
    learning_rate=0.001,
    n_outputs=3,
):
    """
    Build CNN-LSTM hybrid model.
    
    Conv1D extracts local patterns, LSTM captures temporal dependencies,
    Attention pooling focuses on the most relevant time steps.
    """
    inputs = layers.Input(shape=input_shape, name="climate_input")
    
    # CNN feature extraction
    x = inputs
    for i, filters in enumerate(conv_filters):
        x = layers.Conv1D(
            filters, kernel_size, padding="same",
            activation="relu", name=f"conv1d_{i}"
        )(x)
        x = layers.BatchNormalization()(x)
        if i < len(conv_filters) - 1:
            x = layers.Dropout(dropout_rate / 2)(x)
    
    # Max pooling (reduce temporal dimension slightly)
    if input_shape[0] > 6:
        x = layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM for temporal dependencies
    x = layers.LSTM(lstm_units, return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Attention pooling
    x = AttentionPooling(lstm_units, name="attention_pool")(x)
    
    # Dense layers
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Multi-task outputs
    outputs = []
    task_names = ["extreme_heat_prob", "flood_prob", "drought_prob"]
    
    for name in task_names[:n_outputs]:
        head = layers.Dense(16, activation="relu", name=f"{name}_hidden")(x)
        out = layers.Dense(1, activation="sigmoid", name=name)(head)
        outputs.append(out)
    
    if len(outputs) == 1:
        model = Model(inputs=inputs, outputs=outputs[0], name="CNN_LSTM_Hybrid")
    else:
        model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_MultiTask")
    
    if len(outputs) > 1:
        losses = {name: "binary_crossentropy" for name in task_names[:n_outputs]}
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=losses,
            metrics=["mae"],
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )
    
    return model


def train_cnn_lstm(X_train, y_train, X_test, y_test,
                   epochs=100, batch_size=32, output_dir=None):
    """Train CNN-LSTM model."""
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    if y_train.ndim > 1:
        n_outputs = min(y_train.shape[1], 3)
        task_names = ["extreme_heat_prob", "flood_prob", "drought_prob"]
        y_train_dict = {name: y_train[:, i] for i, name in enumerate(task_names[:n_outputs])}
        y_test_dict = {name: y_test[:, i] for i, name in enumerate(task_names[:n_outputs])}
    else:
        n_outputs = 1
        y_train_dict = y_train
        y_test_dict = y_test
    
    model = build_cnn_lstm_model(
        input_shape=input_shape,
        n_outputs=n_outputs,
    )
    
    model.summary()
    
    cb = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    ]
    
    history = model.fit(
        X_train, y_train_dict,
        validation_data=(X_test, y_test_dict),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1,
    )
    
    metrics = model.evaluate(X_test, y_test_dict, verbose=0)
    if isinstance(metrics, list):
        metrics = dict(zip(model.metrics_names, metrics))
    else:
        metrics = {"loss": metrics}
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "cnn_lstm_metrics.json"), "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    
    return model, history, metrics


if __name__ == "__main__":
    print("CNN-LSTM Hybrid Climate Forecasting Module")
    
    np.random.seed(42)
    model = build_cnn_lstm_model(input_shape=(12, 9), n_outputs=3)
    model.summary()
    print(f"\nParameters: {model.count_params():,}")
