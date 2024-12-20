
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Example: Load a time-series dataset
data_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/WHA_mc9FWSVjCGLSLAp48A/stock-prices.csv'
df = pd.read_csv(data_url)

# Select the 'Close' column for training (or any relevant column for your task)
data = df[['Close']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Check the number of samples in the data
n_samples = data.shape[0]
print(f"Number of samples in the dataset: {n_samples}")

# Ensure we have enough data by setting a reasonable train size
if n_samples < 100:
    print("Dataset is very small. Using 50% of the data for training.")
    train_size = 0.5  # Use 50% of data if we have less than 100 samples
    seq_length = 5    # Use a shorter sequence length for very small datasets
else:
    train_size = 0.1  # Use 10% of data for larger datasets
    seq_length = 50   # Change sequence length to 50 for larger datasets

# Reduce dataset size for quicker runs
X, _, Y, _ = train_test_split(data, data, train_size=train_size, random_state=42)  # Adjust train_size based on data

# Preprocess the dataset with adjusted sequence length
def create_dataset(data, time_step=seq_length):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

X, Y = create_dataset(data, seq_length)

# Check if the generated sequences have valid shapes
if X.size == 0 or Y.size == 0:
    raise ValueError(f"The dataset is too small to create sequences with a length of {seq_length}. Reduce the sequence length or use a larger dataset.")

X = X.reshape(X.shape[0], X.shape[1], 1)

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

# Define a simpler Transformer Block for faster runs
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):  # Set training argument default to False
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Build and compile the model
input_shape = (X.shape[1], X.shape[2])
inputs = tf.keras.Input(shape=input_shape)
x = tf.keras.layers.Dense(64)(inputs)  # Reduced embed_dim for faster runs
transformer_block = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128)  # Reduced model complexity
x = transformer_block(x, training=True)  # Pass training argument here
flatten = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(1)(flatten)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='mse')

# Early stopping to stop training when no improvement is seen
early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

# Train the model with reduced epochs and steps
history = model.fit(X, Y, epochs=5, batch_size=32, steps_per_epoch=10, callbacks=[early_stopping])  # Reduced epochs and steps per epoch

# Plot training loss
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()