#!/usr/bin/env python3
"""
climate_transformer.py
======================
Transformer Encoder model for climate time series forecasting.

Architecture:
  Input → Positional Encoding → 
  N × (Multi-Head Attention → Feed-Forward Network → LayerNorm + Residual) →
  Global Average Pooling → Dense → Output

Comparison baseline against LSTM-Attention.
Captures ultra-long-range dependencies in climate sequences.
"""

import numpy as np
import json
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ─── Positional Encoding ────────────────────────────────────────────────────

class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding for time series.
    Injects temporal position information into the input embeddings.
    """
    
    def __init__(self, max_len=512, d_model=64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
        # Precompute positional encodings
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :tf.shape(x)[2]]
    
    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


# ─── Transformer Block ──────────────────────────────────────────────────────

class TransformerBlock(layers.Layer):
    """
    Single Transformer Encoder block.
    Multi-Head Attention → Add & Norm → Feed-Forward → Add & Norm
    """
    
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        # Multi-Head Self-Attention with residual
        attn_output = self.mha(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x1 = self.layernorm1(x + attn_output)
        
        # Feed-Forward Network with residual
        ffn_output = self.ffn(x1)
        ffn_output = self.dropout2(ffn_output, training=training)
        x2 = self.layernorm2(x1 + ffn_output)
        
        return x2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ─── Full Transformer Model ─────────────────────────────────────────────────

def build_transformer_model(
    input_shape,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    num_blocks=3,
    dropout_rate=0.2,
    learning_rate=0.001,
    n_outputs=3,
):
    """
    Build Transformer Encoder model for climate forecasting.
    
    Parameters:
        input_shape: tuple (sequence_length, n_features)
        d_model: dimension of model embeddings
        num_heads: number of attention heads
        ff_dim: feed-forward network dimension
        num_blocks: number of transformer blocks
        dropout_rate: dropout probability
        n_outputs: number of output targets
    """
    inputs = layers.Input(shape=input_shape, name="climate_input")
    
    # Project input features to d_model dimension
    x = layers.Dense(d_model, name="input_projection")(inputs)
    
    # Positional encoding
    x = PositionalEncoding(max_len=input_shape[0], d_model=d_model)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Transformer encoder blocks
    for i in range(num_blocks):
        x = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"transformer_block_{i}"
        )(x)
    
    # Global average pooling over time steps
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(32, activation="relu")(x)
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
        model = Model(inputs=inputs, outputs=outputs[0], name="Transformer_Encoder")
    else:
        model = Model(inputs=inputs, outputs=outputs, name="Transformer_MultiTask")
    
    # Compile
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


# ─── Training ───────────────────────────────────────────────────────────────

def train_transformer(X_train, y_train, X_test, y_test, 
                      epochs=100, batch_size=32, output_dir=None):
    """Train the Transformer model."""
    
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
    
    model = build_transformer_model(
        input_shape=input_shape,
        d_model=64,
        num_heads=4,
        ff_dim=128,
        num_blocks=3,
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
        with open(os.path.join(output_dir, "transformer_metrics.json"), "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    
    return model, history, metrics


if __name__ == "__main__":
    print("Transformer Encoder Climate Forecasting Module")
    
    # Quick test
    np.random.seed(42)
    X = np.random.randn(200, 12, 9)
    
    model = build_transformer_model(
        input_shape=(12, 9), d_model=64, num_heads=4, num_blocks=2, n_outputs=3
    )
    model.summary()
    print(f"\nParameters: {model.count_params():,}")
