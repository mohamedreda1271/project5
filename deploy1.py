from flask import Flask, request, jsonify
import base64
import speech_recognition as sr
import Levenshtein as lev
from io import BytesIO
from pydub import AudioSegment
import ffmpeg
import numpy as np

app = Flask(__name__)

# Constants for error messages
ERROR_MISSING_AUDIO_DATA = "Missing audio data"
ERROR_MISSING_EXPECTED_TEXT = "Missing expected text"
ERROR_INVALID_AUDIO_FORMAT = "Invalid audio format; please provide a PCM WAV, AIFF/AIFF-C, or Native FLAC base64 string"
ERROR_UNKNOWN_VALUE = "Unknown value"
ERROR_REQUEST = "Request error"


def decode_base64_audio(base64_audio):
    """Decode base64 audio data."""
    return base64.b64decode(base64_audio)

def convert_mp3_to_wav(audio_bytes) -> str:
    audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
    audio.export("output.wav", format="wav", parameters=["-ac", "1", "-ar", "16000"])
    return "output.wav"

def recognize_audio_from_bytes(audio_path):
    """Recognize text from audio bytes."""
    recognizer = sr.Recognizer()
    audio_bytes = open(audio_path, 'rb')
    with sr.AudioFile(audio_bytes) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return ERROR_UNKNOWN_VALUE
        except sr.RequestError as e:
            return ERROR_REQUEST.format(e)


def calculate_accuracy(expected_text, recognized_text):
    """Calculate accuracy between expected and recognized text."""
    distance = lev.distance(expected_text.lower(), recognized_text.lower())
    max_len = max(len(expected_text), len(recognized_text))
    return ((max_len - distance) / max_len) * 100


@app.route('/recognize', methods=['POST'])
def recognize_audio():
    """Recognize audio endpoint."""
    data = request.json
    base64_audio_data = data.get('record', '')
    expected_text = data.get('sentence', '')

    if not base64_audio_data:
        return jsonify({'error': ERROR_MISSING_AUDIO_DATA}), 400
    if not expected_text:
        return jsonify({'error': ERROR_MISSING_EXPECTED_TEXT}), 400

    try:
        audio_bytes = decode_base64_audio(base64_audio_data)
        wav_audio_path = convert_mp3_to_wav(audio_bytes)
        recognized_text = recognize_audio_from_bytes(wav_audio_path)
    except ValueError as e:
        return jsonify({'error': ERROR_INVALID_AUDIO_FORMAT, 'details': str(e)}), 400

    if recognized_text in [ERROR_UNKNOWN_VALUE, ERROR_REQUEST]:
        return jsonify({'error': recognized_text}), 400

    accuracy_percentage = calculate_accuracy(expected_text, recognized_text)
    response = {
        'recognized_text': recognized_text,
        'accuracy': accuracy_percentage
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5000)