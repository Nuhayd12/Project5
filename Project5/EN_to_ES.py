import keras_nlp
import pandas as pd
import tensorflow as tf
from keras.layers import TextVectorization
import pathlib
import random
import string
import re
import numpy as np
from tensorflow import keras
from keras import layers
import pickle
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Enable GPU with TensorFlow
if tf.config.list_physical_devices('GPU'):
    print("GPU is available! TensorFlow will use the GPU.")
else:
    print("GPU not detected. Ensure CUDA and cuDNN are installed correctly.")

# Configuration class
class Config:
    vocab_size = 15000  # Vocabulary Size
    sequence_length = 20
    batch_size = 8
    validation_split = 0.15
    embed_dim = 256
    latent_dim = 256
    num_heads = 2
    epochs = 10  # Number of Epochs to train
    start_token = "[start]"
    end_token = "[end]"

config = Config()

# Preprocessing
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "").replace("]", "")

def spanish_standardize(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# Text vectorization
english_vectorization = TextVectorization(
    max_tokens=config.vocab_size,
    output_mode="int",
    output_sequence_length=config.sequence_length,
)
spanish_vectorization = TextVectorization(
    max_tokens=config.vocab_size,
    output_mode="int",
    output_sequence_length=config.sequence_length + 1,
    standardize=spanish_standardize,
)

def preprocess(english, spanish):
    english = english_vectorization(english)
    spanish = spanish_vectorization(spanish)
    return ({"encoder_inputs": english, "decoder_inputs": spanish[:, :-1]}, spanish[:, 1:])

def make_dataset(df, batch_size, mode):
    dataset = tf.data.Dataset.from_tensor_slices((list(df["english"]), list(df["spanish"])))
    if mode == "train":
        dataset = dataset.shuffle(batch_size * 4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(preprocess)
    dataset = dataset.prefetch(tf.data.AUTOTUNE).cache()
    return dataset

# Define the Transformer model
def get_model(config):
    # Encoder
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    encoder_embedding = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=config.vocab_size,
        sequence_length=config.sequence_length,
        embedding_dim=config.embed_dim,
    )
    x = encoder_embedding(encoder_inputs)
    encoder_outputs = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=config.embed_dim,
        num_heads=config.num_heads,
    )(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    # Decoder
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, config.embed_dim), name="encoder_outputs")

    decoder_embedding = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=config.vocab_size,
        sequence_length=config.sequence_length,
        embedding_dim=config.embed_dim,
    )

    x = decoder_embedding(decoder_inputs)

    x = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=config.latent_dim,
        num_heads=config.num_heads,
    )(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
    x = layers.Dropout(0.1)(x)
    decoder_outputs = layers.Dense(config.vocab_size, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    # Connect encoder and decoder
    encoder_outputs = encoder(encoder_inputs)
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])

    # Final transformer model
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )

    transformer.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return transformer

def load_model_and_helpers():
    """Load the pre-trained model and all necessary helpers for translation."""
    global english_vectorization, spanish_vectorization
    global spanish_vocab, spanish_index_lookup
    global start_index, end_index, unk_index

    # Load pre-trained model
    loaded_model = tf.keras.models.load_model("model.keras", custom_objects={
        "TokenAndPositionEmbedding": keras_nlp.layers.TokenAndPositionEmbedding,
        "TransformerEncoder": keras_nlp.layers.TransformerEncoder,
        "TransformerDecoder": keras_nlp.layers.TransformerDecoder,
    })

    # Load vectorization layers' vocabulary from saved files
    with open("english_vocab.pkl", "rb") as f:
        english_vocab = pickle.load(f)
    with open("spanish_vocab.pkl", "rb") as f:
        spanish_vocab = pickle.load(f)

    # Reinitialize vectorization layers with the saved vocab
    english_vectorization.set_vocabulary(english_vocab)
    spanish_vectorization.set_vocabulary(spanish_vocab)

    # Populate Spanish vocabulary lookup
    spanish_index_lookup = dict(zip(range(len(spanish_vocab)), spanish_vocab))

    # Define token indices
    start_index = spanish_vocab.index(config.start_token)
    end_index = spanish_vocab.index(config.end_token)
    unk_index = spanish_vocab.index("[UNK]")

    def decode_sequence(model, input_sentence):
        """Translate an English input sentence to Spanish using the Transformer model."""
        tokenized_input_sentence = english_vectorization([input_sentence])
        decoded_sentence = [start_index]
        for i in range(config.sequence_length):
            decoded_sentence_constant = tf.constant([decoded_sentence[:config.sequence_length]])
            predictions = model([tokenized_input_sentence, decoded_sentence_constant])
            sampled_token_index = np.argmax(predictions[0, i, :])
            decoded_sentence.append(sampled_token_index)
            if sampled_token_index == end_index:
                break
        filtered_tokens = [t for t in decoded_sentence if t not in {start_index, end_index, unk_index}]
        return " ".join([spanish_index_lookup[t] for t in filtered_tokens])

    return loaded_model, decode_sequence

import tensorflow as tf
import keras_nlp
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Load the model
model = tf.keras.models.load_model("model.keras", custom_objects={
    "TokenAndPositionEmbedding": keras_nlp.layers.TokenAndPositionEmbedding,
    "TransformerEncoder": keras_nlp.layers.TransformerEncoder,
    "TransformerDecoder": keras_nlp.layers.TransformerDecoder,
})



# Main block: This runs only when this script is executed directly
if __name__ == "__main__":
    if not os.path.exists("model.keras") or not os.path.exists("english_vocab.pkl") or not os.path.exists("spanish_vocab.pkl"):
        print("Training the model from scratch...")
        # Load the dataset
        data = pd.read_csv("data.csv")
        data["spanish"] = data["spanish"].apply(lambda item: f"{config.start_token} " + item + f" {config.end_token}")

        # Adapt vectorizations
        english_vectorization.adapt(list(data["english"]))
        spanish_vectorization.adapt(list(data["spanish"]))

        # Save vocabularies
        with open("english_vocab.pkl", "wb") as f:
            pickle.dump(english_vectorization.get_vocabulary(), f)
        with open("spanish_vocab.pkl", "wb") as f:
            pickle.dump(spanish_vectorization.get_vocabulary(), f)

        # Split data into train and validation sets
        train, valid = train_test_split(data, test_size=config.validation_split)
        train_ds = make_dataset(train, batch_size=config.batch_size, mode="train")
        valid_ds = make_dataset(valid, batch_size=config.batch_size, mode="valid")

        # Train the model
        model = get_model(config)
        model.summary()

        # Callbacks for training
        model_name = "model.keras"
        checkpoints = tf.keras.callbacks.ModelCheckpoint(
            model_name,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=10,
            monitor="val_loss",
            mode="min",
            restore_best_weights=True,
        )

        # Train the model
        model.fit(
            train_ds, epochs=config.epochs, validation_data=valid_ds, callbacks=[checkpoints, early_stop]
        )
    else:
        print("Model and vocab files already exist. Skipping training...")

    # Load the model and helpers
    loaded_model, decode_sequence = load_model_and_helpers()

    # Test the model with some translations
    data = pd.read_csv("data.csv")
    for i in tqdm(np.random.choice(len(data), 10)):
        item = data.iloc[i]
        translated = decode_sequence(loaded_model, item["english"])
        print("English:", item["english"])
        print("Spanish:", item["spanish"].replace("[start] ", "").replace(" [end]", ""))
        print("Translated:", translated)
