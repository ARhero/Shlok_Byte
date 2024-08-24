from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import PIL
from PIL import Image
import cv2
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

load_dotenv(dotenv_path=".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

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

if __name__ == '__main__':
    app.run(debug=True)
