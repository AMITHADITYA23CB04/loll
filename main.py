import speech_recognition as sr
from transformers import pipeline


# Initialize the recognizer
recognizer = sr.Recognizer()

# Specify the path to your pre-recorded audio file.
# Ensure the path is correctly formatted (using a raw string or escaping backslashes).
audio_file_path = "audio3.wav"

# Open the audio file as the audio source
with sr.AudioFile(audio_file_path) as source:
    # Optionally, adjust for ambient noise (more relevant for live recordings)
    # recognizer.adjust_for_ambient_noise(source, duration=0.2)
    # Record the entire audio file
    audio = recognizer.record(source)

try:
    # Recognize speech using the Google Web Speech API.
    text = recognizer.recognize_google(audio)
    text = text.lower()  # Convert recognized text to lowercase
    print(f"Recognized: {text}")

    # Initialize the sentiment analysis pipeline.
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    # Analyze the sentiment of the recognized text.
    sentiment_result = sentiment_pipeline(text)[0]
    print("Sentiment Analysis Result:", sentiment_result)

    # Optionally, map the sentiment result to an emoji.
    def map_sentiment_to_emoji(sentiment):
        if sentiment.get("label") == "POSITIVE":
            return "😊"
        elif sentiment.get("label") == "NEGATIVE":
            return "😞"
        else:
            return "😐"

    emoji = map_sentiment_to_emoji(sentiment_result)
    print("Your mood is represented by:", emoji)

except sr.UnknownValueError:
    # Handle cases where the speech is unintelligible.
    print("Could not understand the audio. Please try again.")

except sr.RequestError as e:
    # Handle errors related to the API request (e.g., network issues).
    print(f"Could not request results; {e}")
