import tensorflow as tf
import keras_nlp
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import pandas as pd
from keras.layers import TextVectorization
import string
import re
import json
import os
from datetime import datetime
import pickle
from sklearn.model_selection import KFold

# Configuration class
class Config:
    vocab_size = 15000  # Vocabulary Size
    sequence_length = 20
    batch_size = 8
    validation_split = 0.15
    embed_dim = 256
    latent_dim = 256
    num_heads = 2
    epochs = 10
    start_token = "[start]"
    end_token = "[end]"

config = Config()

# Preprocessing setup
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "").replace("]", "")

def spanish_standardize(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# Text vectorization layers
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

# Load the vocabularies
with open("english_vocab.pkl", "rb") as f:
    english_vocab = pickle.load(f)
with open("spanish_vocab.pkl", "rb") as f:
    spanish_vocab = pickle.load(f)

# Set the vocabularies
english_vectorization.set_vocabulary(english_vocab)
spanish_vectorization.set_vocabulary(spanish_vocab)

def preprocess(english, spanish):
    english = english_vectorization(english)
    spanish = spanish_vectorization(spanish)
    return ({"encoder_inputs": english, "decoder_inputs": spanish[:, :-1]}, spanish[:, 1:])

def make_dataset(df, batch_size, mode):
    """
    Create a dataset from a pandas dataframe.
    """
    # Add start and end tokens to Spanish sentences
    df = df.copy()  # Create a copy to avoid modifying the original
    df["spanish"] = df["spanish"].apply(lambda x: f"{config.start_token} {x} {config.end_token}")
    
    # Create tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        list(df["english"]), 
        list(df["spanish"])
    ))
    
    # Shuffle if in training mode
    if mode == "train":
        dataset = dataset.shuffle(batch_size * 4)
    
    # Batch and preprocess
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def evaluate_and_save_metrics(model, test_ds):
    """Evaluate model and save metrics"""
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(test_ds)
    
    # Create metrics dictionary
    metrics = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save metrics
    metrics_file = 'model_metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            history = json.load(f)
            history['evaluations'].append(metrics)
    else:
        history = {'evaluations': [metrics]}
    
    with open(metrics_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    return test_accuracy, test_loss

def evaluate_model_multiple_times(model, data, num_evaluations=5, batch_size=32):
    """Evaluate model multiple times on different subsets of data"""
    kf = KFold(n_splits=num_evaluations, shuffle=True)
    metrics_list = []
    
    print("\nPerforming multiple evaluations...")
    for fold_idx, (_, test_idx) in enumerate(kf.split(data)):
        # Get subset of data
        test_fold = data.iloc[test_idx]
        test_ds = make_dataset(test_fold, batch_size, mode="test")
        
        # Evaluate
        print(f"\nEvaluation {fold_idx + 1}/{num_evaluations}")
        test_loss, test_accuracy = model.evaluate(test_ds)
        
        metrics_list.append({
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'fold': fold_idx
        })
    
    # Save metrics
    metrics_file = 'model_metrics.json'
    history = {'evaluations': metrics_list}
    with open(metrics_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    return metrics_list

def plot_performance_history(metrics_list):
    """Plot performance history from metrics"""
    accuracies = [m['test_accuracy'] for m in metrics_list]
    losses = [m['test_loss'] for m in metrics_list]
    folds = [m['fold'] for m in metrics_list]
    
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(folds, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.title('Model Accuracy Across Evaluations')
    plt.xlabel('Evaluation Number')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim([min(accuracies) - 0.02, max(accuracies) + 0.02])
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(folds, losses, marker='o', color='red', linestyle='-', linewidth=2, markersize=8)
    plt.title('Model Loss Across Evaluations')
    plt.xlabel('Evaluation Number')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.ylim([min(losses) - 0.02, max(losses) + 0.02])
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('performance_history.png')
    plt.show()
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
    print(f"Average Loss: {np.mean(losses):.4f} (±{np.std(losses):.4f})")
    print(f"\nBest Accuracy: {max(accuracies):.4f}")
    print(f"Best Loss: {min(losses):.4f}")
    
if __name__ == "__main__":
    # Load the model
    print("Loading model...")
    model = tf.keras.models.load_model("model.keras", custom_objects={
        "TokenAndPositionEmbedding": keras_nlp.layers.TokenAndPositionEmbedding,
        "TransformerEncoder": keras_nlp.layers.TransformerEncoder,
        "TransformerDecoder": keras_nlp.layers.TransformerDecoder,
    })

    # Create test dataset
    print("Loading and preparing test data...")
    data = pd.read_csv("data.csv")
    test_ds = make_dataset(data.sample(200), batch_size=32, mode="test")

    # Evaluate and save metrics
    test_accuracy, test_loss = evaluate_and_save_metrics(model, test_ds)

    metrics_list = evaluate_model_multiple_times(model, data, num_evaluations=5)
    # Plot performance history
    plot_performance_history(metrics_list)