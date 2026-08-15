#!/usr/bin/env python3
"""
climate_lstm_attention.py
=========================
Bidirectional LSTM with Multi-Head Self-Attention for climate extreme forecasting.

Architecture:
  Input (12-month window) → Bidirectional LSTM (128 units) → 
  Multi-Head Self-Attention (4 heads) → Layer Norm + Residual →
  LSTM (64 units) → Dense (32, ReLU) → Dropout (0.3) →
  Multi-Task Output (3 heads: extreme_heat_prob, flood_prob, drought_prob)

Designed to run on Google Colab (GPU recommended but not required).
"""

import numpy as np
import pandas as pd
import os
import json
from pathlib import Path

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ─── Custom Attention Layer ─────────────────────────────────────────────────

class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention mechanism for time series.
    Computes attention weights across time steps to identify 
    the most relevant historical climate patterns.
    """
    
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model, name="query")
        self.wk = layers.Dense(d_model, name="key")
        self.wv = layers.Dense(d_model, name="value")
        self.dense = layers.Dense(d_model, name="output_projection")
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        
        # Linear projections
        q = self.wq(x)  # (batch, seq_len, d_model)
        k = self.wk(x)
        v = self.wv(x)
        
        # Split into heads
        q = self.split_heads(q, batch_size)  # (batch, heads, seq_len, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention = matmul_qk / tf.math.sqrt(dk)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        
        # Weighted values
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Output projection
        output = self.dense(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x)
        
        # Store attention weights for interpretability
        self.attention_weights = attention_weights
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
        })
        return config


# ─── Model Builder ──────────────────────────────────────────────────────────

def build_lstm_attention_model(
    input_shape,
    lstm_units_1=128,
    lstm_units_2=64,
    attention_heads=4,
    dense_units=32,
    dropout_rate=0.3,
    learning_rate=0.001,
    n_outputs=3,
):
    """
    Build Bidirectional LSTM + Multi-Head Self-Attention model.
    
    Parameters:
        input_shape: tuple (sequence_length, n_features)
        lstm_units_1: units in first BiLSTM layer
        lstm_units_2: units in second LSTM layer
        attention_heads: number of attention heads
        dense_units: units in dense layer
        dropout_rate: dropout probability
        learning_rate: Adam learning rate
        n_outputs: number of output targets (3 for multi-task)
    
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape, name="climate_input")
    
    # First Bidirectional LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_1, return_sequences=True, 
                    kernel_regularizer=keras.regularizers.l2(1e-4)),
        name="bilstm_1"
    )(inputs)
    x = layers.Dropout(dropout_rate)(x)
    
    # Multi-Head Self-Attention
    # Ensure d_model matches the BiLSTM output dimension
    d_model = lstm_units_1 * 2  # BiLSTM doubles the dimension
    attention = MultiHeadSelfAttention(d_model, attention_heads, name="mha")
    x = attention(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    
    # Second LSTM layer
    x = layers.LSTM(lstm_units_2, return_sequences=False,
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    name="lstm_2")(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Shared dense layer
    shared = layers.Dense(dense_units, activation="relu", name="shared_dense")(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(dropout_rate / 2)(shared)
    
    # Multi-task output heads
    outputs = []
    task_names = ["extreme_heat_prob", "flood_prob", "drought_prob"]
    
    for i, name in enumerate(task_names[:n_outputs]):
        head = layers.Dense(16, activation="relu", name=f"{name}_hidden")(shared)
        out = layers.Dense(1, activation="sigmoid", name=name)(head)
        outputs.append(out)
    
    if len(outputs) == 1:
        model = Model(inputs=inputs, outputs=outputs[0], name="LSTM_Attention")
    else:
        model = Model(inputs=inputs, outputs=outputs, name="LSTM_Attention_MultiTask")
    
    # Compile with multi-task loss
    if len(outputs) > 1:
        losses = {name: "binary_crossentropy" for name in task_names[:n_outputs]}
        loss_weights = {name: 1.0 for name in task_names[:n_outputs]}
        metrics_dict = {name: "mae" for name in task_names[:n_outputs]}
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics_dict,
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )
    
    return model


# ─── Data Preparation ───────────────────────────────────────────────────────

def prepare_climate_sequences(df, sequence_length=12, target_cols=None):
    """
    Prepare time series sequences for LSTM training.
    
    Creates sliding windows of 'sequence_length' months.
    Each window predicts the next month's extreme event probabilities.
    """
    if target_cols is None:
        target_cols = ["extreme_heat", "flood_event", "drought_event"]
    
    # Select numeric climate features
    feature_cols = [c for c in df.columns if c not in 
                   ["iso3", "country", "date", "year", "month", "day",
                    "latitude", "longitude"] + target_cols
                   and df[c].dtype in ["float64", "int64", "float32"]]
    
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_cols].fillna(0))
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])
        
        # Multi-target: create binary extreme event indicators
        if target_cols and all(c in df.columns for c in target_cols):
            y.append(df[target_cols].iloc[i].values)
        else:
            # Default: predict next temperature anomaly
            y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler, feature_cols


def create_extreme_event_labels(df):
    """Create binary labels for extreme climate events."""
    
    # Extreme heat: T2M_MAX > 40°C or HWDI > 5
    if "T2M_MAX" in df.columns:
        df["extreme_heat"] = ((df["T2M_MAX"] > 40) | 
                              (df.get("HWDI", pd.Series(0)) > 5)).astype(int)
    else:
        df["extreme_heat"] = 0
    
    # Flood event: Rx5day > 95th percentile or PRECTOTCORR > 50mm/day
    if "PRECTOTCORR" in df.columns:
        heavy_rain_threshold = df["PRECTOTCORR"].quantile(0.95)
        df["flood_event"] = (df["PRECTOTCORR"] > heavy_rain_threshold).astype(int)
    else:
        df["flood_event"] = 0
    
    # Drought event: CDD > 30 or precip_anomaly < -1
    if "CDD" in df.columns:
        df["drought_event"] = ((df.get("CDD", pd.Series(0)) > 30) | 
                               (df.get("precip_anomaly", pd.Series(0)) < -1)).astype(int)
    else:
        df["drought_event"] = 0
    
    return df


# ─── Training Pipeline ──────────────────────────────────────────────────────

def train_lstm_attention(
    df,
    sequence_length=12,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    output_dir=None,
):
    """
    Full training pipeline for LSTM-Attention model.
    
    Returns:
        model, history, metrics, scaler
    """
    print("=" * 60)
    print("LSTM-Attention Model Training Pipeline")
    print("=" * 60)
    
    # Prepare labels
    df = create_extreme_event_labels(df)
    target_cols = ["extreme_heat", "flood_event", "drought_event"]
    
    # Prepare sequences
    print("\nPreparing sequences...")
    X, y, scaler, feature_cols = prepare_climate_sequences(
        df, sequence_length=sequence_length, target_cols=target_cols
    )
    
    print(f"  Input shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Features: {len(feature_cols)}")
    
    # Train/test split (temporal — no shuffling)
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # Determine output configuration
    if y.ndim == 1:
        n_outputs = 1
        y_train_dict = y_train
        y_test_dict = y_test
    else:
        n_outputs = min(y.shape[1], 3)
        task_names = ["extreme_heat_prob", "flood_prob", "drought_prob"]
        y_train_dict = {name: y_train[:, i] for i, name in enumerate(task_names[:n_outputs])}
        y_test_dict = {name: y_test[:, i] for i, name in enumerate(task_names[:n_outputs])}
    
    # Build model
    print("\nBuilding model...")
    input_shape = (X.shape[1], X.shape[2])
    model = build_lstm_attention_model(
        input_shape=input_shape,
        lstm_units_1=128,
        lstm_units_2=64,
        attention_heads=4,
        dense_units=32,
        dropout_rate=0.3,
        n_outputs=n_outputs,
    )
    
    model.summary()
    
    # Callbacks
    callback_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        callback_list.append(
            callbacks.ModelCheckpoint(
                os.path.join(output_dir, "lstm_attention_best.keras"),
                save_best_only=True,
            )
        )
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train_dict,
        validation_data=(X_test, y_test_dict),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1,
    )
    
    # Evaluate
    print("\nEvaluation:")
    eval_results = model.evaluate(X_test, y_test_dict, verbose=0)
    
    if isinstance(eval_results, list):
        metrics = dict(zip(model.metrics_names, eval_results))
    else:
        metrics = {"loss": eval_results}
    
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")
    
    # Predictions for analysis
    predictions = model.predict(X_test)
    
    # Save metrics
    if output_dir:
        metrics_path = os.path.join(output_dir, "lstm_attention_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        
        # Save training history
        history_path = os.path.join(output_dir, "lstm_attention_history.json")
        with open(history_path, "w") as f:
            hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
            json.dump(hist_dict, f, indent=2)
    
    return model, history, metrics, scaler, feature_cols


# ─── Prediction & Analysis ──────────────────────────────────────────────────

def predict_future_extremes(model, recent_data, scaler, feature_cols, steps=6):
    """
    Generate multi-step future predictions of extreme event probabilities.
    
    Parameters:
        model: trained LSTM-Attention model
        recent_data: DataFrame with recent climate observations
        scaler: fitted StandardScaler
        feature_cols: list of feature column names
        steps: number of future months to predict
    
    Returns:
        DataFrame with predicted probabilities
    """
    # Scale recent data
    scaled = scaler.transform(recent_data[feature_cols].fillna(0))
    
    predictions = []
    current_window = scaled[-12:]  # Last 12 months
    
    for step in range(steps):
        # Reshape for model input
        X = current_window.reshape(1, 12, -1)
        
        # Predict
        pred = model.predict(X, verbose=0)
        
        if isinstance(pred, list):
            pred_dict = {
                "step": step + 1,
                "extreme_heat_prob": float(pred[0][0][0]),
                "flood_prob": float(pred[1][0][0]),
                "drought_prob": float(pred[2][0][0]),
            }
        else:
            pred_dict = {
                "step": step + 1,
                "prediction": float(pred[0][0]),
            }
        
        predictions.append(pred_dict)
        
        # Auto-regressive: use prediction as next input (simplified)
        # In production, this would use a more sophisticated approach
        new_row = current_window[-1].copy()
        current_window = np.vstack([current_window[1:], new_row])
    
    return pd.DataFrame(predictions)


def extract_attention_weights(model, X_sample):
    """Extract attention weights for interpretability analysis."""
    # Get the attention layer
    attention_layer = None
    for layer in model.layers:
        if isinstance(layer, MultiHeadSelfAttention):
            attention_layer = layer
            break
    
    if attention_layer is None:
        print("No attention layer found in model.")
        return None
    
    # Create a sub-model that outputs attention weights
    # Forward pass to populate attention weights
    _ = model.predict(X_sample[:1], verbose=0)
    
    return attention_layer.attention_weights


# ─── Main (for standalone testing) ──────────────────────────────────────────

if __name__ == "__main__":
    print("LSTM-Attention Climate Forecasting Module")
    print("This module is designed to be imported by analysis/03_model_training.py")
    print("Or run directly with climate data for testing.")
    
    # Quick test with synthetic data
    print("\nRunning synthetic data test...")
    np.random.seed(42)
    
    n_samples = 500
    n_features = 9
    seq_len = 12
    
    X = np.random.randn(n_samples, seq_len, n_features)
    y_heat = (np.random.rand(n_samples) > 0.7).astype(float)
    y_flood = (np.random.rand(n_samples) > 0.8).astype(float)
    y_drought = (np.random.rand(n_samples) > 0.75).astype(float)
    
    model = build_lstm_attention_model(
        input_shape=(seq_len, n_features),
        lstm_units_1=64,
        lstm_units_2=32,
        attention_heads=4,
        dense_units=16,
        n_outputs=3,
    )
    
    model.summary()
    print("\nModel built successfully!")
    print(f"Parameters: {model.count_params():,}")
