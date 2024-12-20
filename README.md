Build Transformers Model for Text Generation with TensorFlow
============================================================

Imagine sitting down to write a poem or story, and instead of struggling for words, a machine helps you out. This is what **Transformer models** do in the world of **Natural Language Processing (NLP)**. They can generate text that sounds human-like, create complete sentences, and even mimic creative writing styles like **Shakespearean poetry** or code. If you're curious about how to set up such a model and start generating text, this guide will walk you through the steps of building a **Transformer model for text generation** using **TensorFlow**.

### Why Should You Care About Text Generation with Transformers?

Think about how **AI** like **ChatGPT** or **GPT-3** can generate text that feels so natural, coherent, and creative. That magic happens thanks to **Transformer models**. Transformers have revolutionized **Natural Language Processing (NLP)** and are now a cornerstone of **deep learning**. But what makes them so powerful?

1.  **Capturing Context**: Transformers can remember information from previous parts of the text and use it to generate coherent sentences.
    
2.  **Understanding Relationships**: They model complex relationships between words or characters, even if they are far apart in the text.
    
3.  **Versatility**: Transformers are not limited to text data. They have been successfully applied in other fields such as **computer vision**, **speech recognition**, and even **reinforcement learning**.
    

By understanding and implementing Transformers, you can build powerful models for a variety of tasks, from text generation to time series forecasting and beyond.

### The Transformer Architecture: Encoder and Decoder

A **Transformer model** consists of two main parts: the **encoder** and the **decoder**. Both components are composed of layers that include:

*   **Self-attention mechanisms**: These allow the model to focus on different parts of the sequence, capturing relationships between distant tokens.
    
*   **Feedforward neural networks**: These are used to process the information from the attention mechanisms and help the model learn complex patterns.
    

The **self-attention** mechanism in the Transformer enables the model to attend to all positions in the input sequence simultaneously. This ability to capture dependencies in sequential data, regardless of their position in the sequence, addresses the limitations of older models like **RNNs** and **LSTMs**.

### Why Transformers Are a Game Changer for Sequential Data

Sequential data, like text, time series, and speech, has an inherent order, with each element depending on the ones that came before it. In traditional models like **RNNs** and **LSTMs**, this order is processed step-by-step. However, these models have limitations, particularly when dealing with long-range dependencies, because they process data sequentially.

Transformers solve this problem with **self-attention**, allowing the model to focus on all positions in the sequence at once, rather than one by one. This makes them much more efficient and effective at learning from long sequences of data.

### Step 1: Setting Up the Environment

Before diving into the code, let’s make sure your environment is ready. Start by installing **TensorFlow** and the necessary libraries:

````
!pip install tensorflow==2.16.2

````

Then, let’s import the libraries we’ll be using:
````
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import get_file
import matplotlib.pyplot as plt

````
### Step 2: Loading and Preprocessing the Dataset

### Loading Shakespeare’s Text


We’ll use the **Shakespeare dataset** for this guide, as it will help us generate text in a Shakespearean style. You can replace this with any other dataset for text generation.

````
path_to_file = get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

````

### Preprocessing the Text

Next, we preprocess the text so that the model can work with it. We’ll use **TextVectorization** to convert the text into integer sequences, where each word is represented by a unique number.

# Setting the maximum vocabulary size and sequence length
````
vocab_size = 10000  # The maximum number of unique tokens (words)
seq_length = 100  # Length of each sequence for training

````
# Vectorizing the text using TextVectorization
````
vectorizer = TextVectorization(max_tokens=vocab_size, output_mode='int')
text_ds = tf.data.Dataset.from_tensor_slices([text]).batch(1)
vectorizer.adapt(text_ds)
````

# Vectorizing the text
````
vectorized_text = vectorizer([text])[0]
print("Vectorized text shape:", vectorized_text.shape)
print("First 10 vectorized tokens:", vectorized_text.numpy()[:10])
````

### Step 3: Creating Sequences for Training

To generate text, we need to prepare **input-output sequences**. Each input sequence will consist of 100 consecutive tokens (words), and the output sequence will be the next token after those 100.

````
def create_sequences(text, seq_length): 
    input_seqs = [] 
    target_seqs = [] 
    for i in range(len(text) - seq_length): 
        input_seq = text[i:i + seq_length] 
        target_seq = text[i + 1:i + seq_length + 1] 
        input_seqs.append(input_seq) 
        target_seqs.append(target_seq) 
    return np.array(input_seqs), np.array(target_seqs)

X, Y = create_sequences(vectorized_text.numpy(), seq_length)
````

### Step 4: Building the Transformer Model

Now, let’s define the **Transformer model**. A Transformer block consists of **multi-head attention mechanisms** and **feed-forward networks**, which allow it to focus on different parts of the sequence while generating text.

### Defining the Transformer Block
````
class TransformerBlock(tf.keras.layers.Layer):
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

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
````
### Building the Full Transformer Model

````
class TransformerModel(Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(seq_length, embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.dense = Dense(vocab_size)

    def positional_encoding(self, seq_length, embed_dim):
        angle_rads = self.get_angles(np.arange(seq_length)[:, np.newaxis], np.arange(embed_dim)[np.newaxis, :], embed_dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, embed_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        x += self.pos_encoding[:, :seq_len, :]
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        output = self.dense(x)
        return output
````
### Step 5: Training the Model

Let’s now train the model. We use **early stopping** to prevent overfitting and monitor the training process.

````
early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

history = model.fit(X, Y, epochs=20, batch_size=32, callbacks=[early_stopping])
````

### Step 6: Visualizing the Training Loss

Visualizing the training loss helps us track how well the model is learning.
````
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
````

### Step 7: Text Generation with the Trained Model

After training, you can generate text using the trained model. Provide a **seed phrase**, and the model will generate additional text based on what it has learned.

````

def generate_text(model, start_string, num_generate=100, temperature=1.0):
    # Convert the start string to a vectorized format
    input_eval = vectorizer([start_string]).numpy()

    # Adjust input length to match model input shape
    if input_eval.shape[1] < seq_length:
        padding = np.zeros((1, seq_length - input_eval.shape[1]))
        input_eval = np.concatenate((padding, input_eval), axis=1)
    elif input_eval.shape[1] > seq_length:
        input_eval = input_eval[:, -seq_length:]

    input_eval = tf.convert_to_tensor(input_eval)
    text_generated = []

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[0]  # Shape: [vocab_size]
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()

        # Update input tensor and generate next word
        input_eval = np.append(input_eval.numpy(), [[predicted_id]], axis=1)
        input_eval = input_eval[:, -seq_length:]
        input_eval = tf.convert_to_tensor(input_eval)
        text_generated.append(vectorizer.get_vocabulary()[predicted_id])

    return start_string + ' ' + ' '.join(text_generated)

````

### Step 8: Generate Text

Generate new text by giving the model a starting string:
````
start_string = "To be, or not to be"
generated_text = generate_text(model, start_string, temperature=0.7)
print(generated_text)
````
### Conclusion

You’ve now built and trained a **Transformer-based model for text generation**. By following these steps, you learned how to:

*   **Preprocess and tokenize text**.
    
*   **Build and train a Transformer model** using **TensorFlow**.
    
*   **Generate new text** based on a given starting string.
    

Transformers are powerful tools for various text generation tasks. They can be applied to **creative writing**, **chatbots**, **time series forecasting**, and even fields outside of NLP, like **computer vision** and **reinforcement learning**.

What will you generate with your Transformer model? Feel free to share your results or ask any questions you might have about text generation!

Feel free to reach out with questions or share your experiences with text generation. Happy coding! 🚀