import gradio as gr
import whisper
from transformers import pipeline

# For recording voice
import sounddevice as sd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

model = whisper.load_model("base")

sentiment_analysis = pipeline(
    "sentiment-analysis",
    framework="pt",
    model="SamLowe/roberta-base-go_emotions"
)

# Text Input --> Sentiment Output
def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {
        result['label']: result['score'] for result in results
    }
    return sentiment_results

# def get_sentiment_emoji(sentiment):
#     # Define the mapping of sentiments to emojis
#     emoji_mapping = {
#         "disappointment": "😞",
#         "sadness": "😢",
#         "annoyance": "😠",
#         "neutral": "😐",
#         "disapproval": "👎",
#         "realization": "😮",
#         "nervousness": "😬",
#         "approval": "👍",
#         "joy": "😄",
#         "anger": "😡",
#         "embarrassment": "😳",
#         "caring": "🤗",
#         "remorse": "😔",
#         "disgust": "🤢",
#         "grief": "😥",
#         "confusion": "😕",
#         "relief": "😌",
#         "desire": "😍",
#         "admiration": "😌",
#         "optimism": "😊",
#         "fear": "😨",
#         "love": "❤️",
#         "excitement": "🎉",
#         "curiosity": "🤔",
#         "amusement": "😄",
#         "surprise": "😲",
#         "gratitude": "🙏",
#         "pride": "🦁"
#     }
#     return emoji_mapping.get(sentiment, "")

# def display_sentiment_results(sentiment_results, option="Sentiment Only"):
#     sentiment_text = ""
#     for sentiment, score in sentiment_results.items():
#         emoji = get_sentiment_emoji(sentiment)
#         if option == "Sentiment Only":
#             sentiment_text += f"{sentiment} {emoji}\n"
#         elif option == "Sentiment + Score":
#             sentiment_text += f"{sentiment} {emoji}: {score}\n"

#     return sentiment_text

def inference(audio, sentiment_option):
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    sentiment_results = analyze_sentiment(result.text)

    return sentiment_results
    # return lang.upper(), result.text, sentiment_output



if __name__ == "__main__":
    # Recording parameters
    duration = 10  # seconds
    sample_rate = 16000  # Whisper expects 16000 Hz
    channels = 1  # Mono

    # Record audio
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete!")

    # Whisper expects the input as a 1D array
    audio_data = audio_data.flatten()

    print(inference(audio_data,"Sentiment Only"))

    # print(analyze_sentiment("Rohan is good girl. I hate good girls. I am not happy with rohan."))