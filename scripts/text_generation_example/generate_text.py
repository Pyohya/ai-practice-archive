import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
import re

# --- Parameters ---
SEQ_LENGTH = 50 # Length of input sequences
EPOCHS = 150 # Number of epochs to train for
BATCH_SIZE = 128
LSTM_UNITS = 128 # Number of units in the LSTM layer
EMBEDDING_DIM = 64 # Dimension of the character embedding

# --- 1. Load Data ---
def load_corpus(filepath="scripts/text_generation_example/corpus.txt"):
    if not os.path.exists(filepath):
        print(f"Error: Corpus file not found at {filepath}")
        # Create a dummy corpus if not found, to prevent script from crashing
        # In a real scenario, you'd handle this more robustly
        dummy_text = "이것은 예제 텍스트입니다. 모델 학습을 위한 기본 내용입니다."
        print(f"Using a dummy corpus for demonstration: '{dummy_text}'")
        return dummy_text
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    # Simple text cleaning: remove extra newlines and leading/trailing spaces
    text = re.sub(r'\n+', '\n', text).strip()
    print(f"Corpus loaded. Length: {len(text)} characters.")
    return text

# --- 2. Preprocessing ---
def preprocess_text(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Unique characters: {vocab_size} -> {''.join(chars)}")
    
    char_to_int = {char: i for i, char in enumerate(chars)}
    int_to_char = {i: char for i, char in enumerate(chars)}
    
    return char_to_int, int_to_char, chars, vocab_size

# --- 3. Prepare Training Data ---
def prepare_training_data(text, char_to_int, seq_length, vocab_size):
    dataX = []
    dataY = []
    
    for i in range(0, len(text) - seq_length, 1):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
        
    n_patterns = len(dataX)
    print(f"Total patterns: {n_patterns}")
    
    # X should be [samples, time steps] for the Embedding layer
    X = np.array(dataX) # Shape: (n_patterns, seq_length)
    
    # Normalize X (optional, but can help if not using embedding, or for certain activations)
    # X = X / float(vocab_size) # Not needed if using Embedding layer with integer inputs
    
    # One-hot encode the output variable y if using categorical_crossentropy
    # y = tf.keras.utils.to_categorical(dataY, num_classes=vocab_size)
    # For sparse_categorical_crossentropy, y can remain as integer labels
    y = np.array(dataY)
    
    return X, y, n_patterns

# --- 4. Define Model ---
def build_model(seq_length, vocab_size, embedding_dim=EMBEDDING_DIM, lstm_units_param=LSTM_UNITS):
    # lstm_units_param from function signature can be overridden if needed,
    # but here we'll use a fixed value of 256 for the layers as per modification request.
    model_lstm_units = 256 

    model = Sequential([
        # Embedding layer maps each character index to a dense vector of embedding_dim dimensions.
        # Input to Embedding is (batch_size, seq_length)
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length),
        
        # First LSTM layer
        LSTM(model_lstm_units, return_sequences=True), # return_sequences=True because the next layer is another LSTM
        Dropout(0.2), # Dropout for regularization
        
        # Second LSTM layer
        LSTM(model_lstm_units), # No return_sequences=True as this is the last LSTM layer before Dense
        Dropout(0.2), # Dropout for regularization
        
        # Output layer
        Dense(vocab_size, activation='softmax')
    ])
    
    # Using sparse_categorical_crossentropy because y is integer-encoded.
    # If y were one-hot encoded, use categorical_crossentropy.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# --- 5. Text Generation Function ---
def generate_text(model, char_to_int, int_to_char, vocab_size, seq_length, seed_text, n_chars_to_generate):
    print(f"\n--- Generating text with seed: '{seed_text}' ---")
    generated_text = seed_text
    current_sequence_int = [char_to_int.get(char, 0) for char in seed_text] # Use 0 for unknown chars
    
    for _ in range(n_chars_to_generate):
        if len(current_sequence_int) < seq_length:
            # Pad with 0 (or a specific padding character's int value) if shorter
            padded_sequence = [0] * (seq_length - len(current_sequence_int)) + current_sequence_int
        else:
            # Take the last seq_length characters
            padded_sequence = current_sequence_int[-seq_length:]
            
        # Reshape for model input: [1, seq_length] (batch_size, timesteps)
        input_seq_reshaped = np.reshape(padded_sequence, (1, seq_length))
        # input_seq_normalized = input_seq_reshaped / float(vocab_size) # Not needed for Embedding layer

        prediction = model.predict(input_seq_reshaped, verbose=0)[0] # Get probabilities for the next char
        
        # Sample the next character index based on the predicted probabilities
        # Using tf.random.categorical for sampling
        # This avoids always picking the character with the highest probability (argmax), leading to more diverse text.
        # Temperature can be added here to control randomness (not implemented for simplicity)
        next_index = tf.random.categorical(tf.math.log([prediction]), num_samples=1)[0,0].numpy()
        # Alternative: np.random.choice(len(prediction), p=prediction)
        
        next_char = int_to_char.get(next_index, '?') # Use '?' for unknown index
        
        generated_text += next_char
        current_sequence_int.append(next_index)
        
    return generated_text

# --- 6. Main Execution Block ---
if __name__ == '__main__':
    print("--- Starting Character-Level Text Generation Example ---")
    
    # Load data
    corpus_text = load_corpus()
    if not corpus_text:
        print("Exiting due to corpus loading issue.")
        exit()
        
    # Preprocess text
    char_to_int, int_to_char, chars, vocab_size = preprocess_text(corpus_text)
    
    # Prepare training data
    X_train, y_train, n_patterns = prepare_training_data(corpus_text, char_to_int, SEQ_LENGTH, vocab_size)
    
    if n_patterns == 0:
        print(f"Not enough data to create sequences with SEQ_LENGTH={SEQ_LENGTH}. Try a shorter corpus or smaller SEQ_LENGTH.")
        exit()

    # Build model
    model = build_model(SEQ_LENGTH, vocab_size)
    
    # Define callbacks (optional, but good for longer training)
    # filepath_checkpoint = "text_gen_model_checkpoint.weights.h5"
    # checkpoint = ModelCheckpoint(filepath_checkpoint, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)
    # callbacks_list = [checkpoint, reduce_lr]
    callbacks_list = [] # Not using for this simple example to avoid file I/O issues in restricted envs

    print("\n--- Training Model ---")
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list, verbose=2)
    
    # (Optional) Load weights if you saved them with ModelCheckpoint
    # model.load_weights(filepath_checkpoint)
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    # Generate text
    # Pick a random seed from the corpus to start generation
    start_index = np.random.randint(0, len(corpus_text) - SEQ_LENGTH -1)
    if start_index + SEQ_LENGTH > len(corpus_text): # Ensure seed is within bounds
        start_index = 0
    
    seed_text_corpus = corpus_text[start_index : start_index + SEQ_LENGTH]
    
    # Or use a fixed seed
    custom_seed = "호랑이는 엄마에게" 
    if len(custom_seed) < SEQ_LENGTH:
      print(f"\nNote: Custom seed '{custom_seed}' is shorter than SEQ_LENGTH ({SEQ_LENGTH}). It will be padded for initial prediction.")
    
    # Generate text using the corpus-based seed
    generated_sequence_corpus = generate_text(model, char_to_int, int_to_char, vocab_size, SEQ_LENGTH, seed_text_corpus, 200)
    print("\n--- Generated Text (from corpus seed) ---")
    print(generated_sequence_corpus)

    # Generate text using the custom seed
    generated_sequence_custom = generate_text(model, char_to_int, int_to_char, vocab_size, SEQ_LENGTH, custom_seed, 200)
    print("\n--- Generated Text (from custom seed) ---")
    print(generated_sequence_custom)

    print("\n--- Text Generation Example Finished ---")
