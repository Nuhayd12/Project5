import tensorflow as tf
import pickle
import numpy as np
import pyttsx3
import speech_recognition as sr
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerEncoder, TransformerDecoder
from keras.layers import TextVectorization


class Translator:
    def __init__(self, model_path, english_vocab_path, spanish_vocab_path, config):
        self.config = config

        # Load the pre-trained model
        self.model = tf.keras.models.load_model(model_path, custom_objects={
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerEncoder": TransformerEncoder,
            "TransformerDecoder": TransformerDecoder,
        })

        # Load vocabularies
        with open(english_vocab_path, "rb") as f:
            self.english_vocab = pickle.load(f)
        with open(spanish_vocab_path, "rb") as f:
            self.spanish_vocab = pickle.load(f)

        # Initialize vectorization layers
        self.english_vectorization = TextVectorization(
            max_tokens=self.config.vocab_size,
            output_mode="int",
            output_sequence_length=self.config.sequence_length,
        )
        self.english_vectorization.set_vocabulary(self.english_vocab)

        self.spanish_vectorization = TextVectorization(
            max_tokens=self.config.vocab_size,
            output_mode="int",
            output_sequence_length=self.config.sequence_length + 1,
        )
        self.spanish_vectorization.set_vocabulary(self.spanish_vocab)

        # Spanish index-to-word mapping
        self.spanish_index_lookup = {i: word for i, word in enumerate(self.spanish_vocab)}

        # Token indices
        self.start_index = self.spanish_vocab.index(config.start_token)
        self.end_index = self.spanish_vocab.index(config.end_token)

    def translate(self, input_sentence):
        """Translate English text to Spanish."""
        # Vectorize the input English sentence
        tokenized_input = self.english_vectorization([input_sentence])
        
        # Start translation with the start token
        decoded_sentence = [self.start_index]

        for _ in range(self.config.sequence_length):
            # Prepare the decoder input (partial sentence)
            decoder_input = tf.constant([decoded_sentence], dtype=tf.int32)
            
            # Get predictions from the model
            predictions = self.model([tokenized_input, decoder_input])
            
            # Select the token with the highest probability
            sampled_token_index = np.argmax(predictions[0, len(decoded_sentence) - 1, :])
            decoded_sentence.append(sampled_token_index)

            # Stop if the end token is generated
            if sampled_token_index == self.end_index:
                break

        # Convert token indices to words, excluding special tokens
        translated_words = [
            self.spanish_index_lookup[token]
            for token in decoded_sentence
            if token not in {self.start_index, self.end_index}
        ]
        return " ".join(translated_words)


# Configurations for the Transformer model
class Config:
    vocab_size = 15000
    sequence_length = 20
    batch_size = 8
    embed_dim = 256
    latent_dim = 256
    num_heads = 2
    start_token = "[start]"
    end_token = "[end]"


def speak(text):
    """Convert text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def listen():
    """Capture real-time speech input and return as text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except sr.WaitTimeoutError:
            print("Bot: No speech detected, please try again.")
            return ""
        except sr.UnknownValueError:
            print("Bot: Sorry, I didn't understand that.")
            return ""


def run_real_time_voice_translation():
    """Run a real-time voice-based English-to-Spanish Translator."""
    config = Config()
    translator = Translator("model.keras", "english_vocab.pkl", "spanish_vocab.pkl", config)

    print("\nWelcome to the Voice-Based English-to-Spanish Translator!")
    print("Speak a sentence in English, and I'll translate it to Spanish.")
    print("Say 'quit' to exit the conversation.\n")

    while True:
        user_input = listen().strip()
        if user_input.lower() == "quit":
            print("Goodbye!")
            speak("Goodbye!")
            break

        if user_input == "":
            continue

        # Translate the input
        translation = translator.translate(user_input)
        print(f"Bot (Translated to Spanish): {translation}")
        speak(translation)


if __name__ == "__main__":
    run_real_time_voice_translation()
