import speech_recognition as sr
import pyttsx3
import asyncio
from googletrans import Translator  # Ensure this is 'googletrans==4.0.0-rc1'

# Initialize the Translator and Text-to-Speech engine
translator = Translator()
tts_engine = pyttsx3.init()

# Function for text-to-speech output
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Async function for real-time translation
async def real_time_translation():
    recognizer = sr.Recognizer()
    
    print("Speak something in English, and I will translate it to Spanish...")
    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise and listen to the user
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)

            # Convert speech to text
            english_text = recognizer.recognize_google(audio)
            print(f"You said (in English): {english_text}")
            
            # Use the translator asynchronously
            translated = await translator.translate(english_text, src='en', dest='es')
            spanish_text = translated.text
            print(f"Translation (in Spanish): {spanish_text}")
            
            # Use text-to-speech to say the translation
            speak_text(spanish_text)

    except sr.UnknownValueError:
        print("Sorry, I could not understand that.")
    except sr.RequestError as e:
        print(f"Error with the Speech Recognition service: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the main function
asyncio.run(real_time_translation())
