from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import PIL
from PIL import Image
import cv2
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import speech_recognition as sr
from gtts import gTTS
import base64
from googletrans import Translator

app = Flask(__name__)

load_dotenv(dotenv_path=".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

recognizer = sr.Recognizer()

csv_file = 'data/book.csv'
df = pd.read_csv(csv_file, dtype={'Verse': str})
llm = ChatGoogleGenerativeAI(model="gemini-pro")
gif_dest = "alphabet/"

def func(text):
    all_frames = []
    words = text.split()

    for word in words:
        for letter in word:
                gif_file = os.path.join(gif_dest, f"{letter.lower()}_small.png")
                if os.path.exists(gif_file):
                    im = PIL.Image.open(gif_file)
                    frame_count = im.n_frames
                    for frame in range(frame_count):
                        im.seek(frame)
                        im.save("tmp.png")
                        img = cv2.imread("tmp.png")
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (380, 260))
                        im_arr = PIL.Image.fromarray(img)
                        all_frames.append(im_arr)
                    for _ in range(int(0.5 * 10)):
                        all_frames.append(im_arr)
        if all_frames:
            for _ in range(int(0.9 * 10)):
                all_frames.append(im_arr)

    final_gif_path = "static/out.gif"
    all_frames[0].save(final_gif_path, save_all=True, append_images=all_frames[1:], duration=100, loop=0)
    return final_gif_path

# Function to detect the language of the given text
def detect_language(text):
    translator = Translator()
    detected_lang = translator.detect(text).lang
    return detected_lang

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio_data = recognizer.listen(source)
        print("Recognizing...")
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Sorry, the service is down."

@app.route('/')
def index():
    chapters = df['Chapter'].unique().tolist()
    chapters_and_verses = {chapter: df[df['Chapter'] == chapter]['Verse'].nunique() for chapter in chapters}
    return render_template('index.html', chapters_and_verses=chapters_and_verses)

@app.route('/generate_gif', methods=['POST'])
def generate_gif():
    data = request.json
    print(data)
    chapter = data['chapter']
    verse = data['verse']
    
    filtered_df = df[df['Verse'] == f"{chapter}.{verse}"]
    if not filtered_df.empty:
        sanskrit_anuvad = filtered_df['Sanskrit Anuvad'].values[0]
        hindi_anuvad = filtered_df['Hindi Anuvad'].values[0]
        english_translation = filtered_df['English Translation'].values[0]

        text = english_translation  # Use English translation for GIF creation
        gif_path = func(text)
        
        return jsonify({
            'gif_path': gif_path,
            'sanskrit_anuvad': sanskrit_anuvad,
            'hindi_anuvad': hindi_anuvad,
            'english_translation': english_translation
        })
    else:
        return jsonify({'error': 'No matching verse found.'}), 404
    
@app.route('/record', methods=['POST'])
def record():
    if request.method == 'POST':
        recognized_text = recognize_speech()
        ch_num_prompt = "Extract the chapter number from the following text."
        ch_num_output_prompt = "Expected Output: <chapter_number> "
        ver_num_prompt = "Extract the verse number from the following text."
        ver_num_output_prompt = "Expected Output: <verse_number>"
        text_prompt = f"Text: {recognized_text}"

        ch_prompt = ch_num_prompt + "\n" + text_prompt + "\n" + ch_num_output_prompt
        ver_prompt = ver_num_prompt + "\n" + text_prompt + "\n" + ver_num_output_prompt
        
        result = llm.invoke(ch_prompt)
        chapter = result.content
        result = llm.invoke(ver_prompt)
        verse = result.content
        print(chapter, verse)
        filtered_df = df[df['Verse'] == f"{chapter}.{verse}"]
        if not filtered_df.empty:
            hindi_anuvad = filtered_df['English Translation'].values[0]
            print(hindi_anuvad)
            # TTS = default_tts()
            # audio = TTS.synthesize(sanskrit_anuvad)
            # # Export the audio as an MP3
            # audio.export("sanskrit_speech.mp3")
            # tts.text_to_speech(hindi_anuvad, debug=True, use_pronunciation_dict=True)
            
            

            # CHUNK_SIZE = 1024
            # url = "https://api.elevenlabs.io/v1/text-to-speech/Xb7hH8MSUJpSbSDYk0k2"

            # headers = {
            # "Accept": "audio/mpeg",
            # "Content-Type": "application/json",
            # "xi-api-key": "sk_b7bc80749463affef4827223df9088be1a7fc995d8ec5985"
            # }

            # data = {
            # "text": hindi_anuvad,
            # "model_id": "eleven_monolingual_v1",
            # "voice_settings": {
            #     "stability": 0.5,
            #     "similarity_boost": 0.5
            # }
            # }

            # response = requests.post(url, json=data, headers=headers)
            # with open('output.mp3', 'wb') as f:
            #     for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            #         if chunk:
            #             f.write(chunk)
            words = hindi_anuvad.split(' ')
            
            with open('output.mp3', 'wb') as ff:
                for word in words:
                    # Detect the language of the text
                    detected_language = detect_language(word)
                    print(detected_language)
                    gTTS(text=word, lang=detected_language, slow=False).write_to_fp(ff)
            with open("output.mp3", 'rb') as file:
                audio_bytes = file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()
            
        return jsonify({'text': recognized_text,'audio_base64': audio_base64})

@app.route('/generate_response', methods=['POST'])
def generate_response():
    data = request.get_json()
    print(data)
    query = data.get('text')
    print(query)
    qa_prompt = "suggest a solution based on Bhagavad Gita. Also mention the related shloka in sanskrit and english if any. If the question is not related to life issues, then reply 'Sorry, I can only answer questions related to life, not related to (the category of question asked if not life)'. Generate the response and give one line gap after each line if any, dont use * and double quote .----------------"
    input_text = qa_prompt + "\nUser question:\n" + query
    
    # Invoke the Gemini API
    result = llm.invoke(input_text)
    response_text = result.content
    print(result.content)
    return jsonify({'response': response_text})
if __name__ == '__main__':
    app.run(debug=True)
