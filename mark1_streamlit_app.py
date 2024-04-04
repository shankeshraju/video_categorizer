from streamlit_extras.grid import grid
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import cv2
import torch
import shutil
import sqlite3
import clip
import os

st.set_page_config(layout="wide")

if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocess' not in st.session_state:
    st.session_state.preprocess = None
if 'imageVectors' not in st.session_state:
    st.session_state.imageVectors = None
if 'input_file_path' not in st.session_state:
    st.session_state.input_file_path = None
if 'numpy_tensor' not in st.session_state:
    st.session_state.numpy_tensor = None



device = "cpu"
conn = sqlite3.connect("streamlit.db")
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS text_vector
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
               file_name TEXT,
               topic TEXT,
               file_location TEXT,
               vectors BLOB)
               ''')
conn.commit()
categories = ["sports", "entertainment", "politics", "science and technology", "travel"]

@st.cache_resource
def Initialization():
    st.session_state.model, st.session_state.preprocess = clip.load("ViT-B/32", device=device)


def similarity_score(query):
    query_vector = st.session_state.model.encode_text(clip.tokenize([query]).to(device))
    similarity_score = torch.cosine_similarity(st.session_state.imageVectors, query_vector)
    similarity_score = similarity_score.detach().numpy()[0]
    return similarity_score

def similarity_score_query(category, query):
    query_vector = st.session_state.model.encode_text(clip.tokenize([query]).to(device))
    category_vector = st.session_state.model.encode_text(clip.tokenize([category]).to(device))
    similarity_score = torch.cosine_similarity(category_vector, query_vector)
    similarity_score = similarity_score.detach().numpy()[0]
    return similarity_score

def process_video(input_file):
    temp_dir = tempfile.mkdtemp()
    st.session_state.input_file_path = os.path.join(temp_dir, input_file.name)
    with open(st.session_state.input_file_path, "wb") as f:
        f.write(input_file.getvalue())

    cap = cv2.VideoCapture(str(st.session_state.input_file_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"{frame_count = }")
    print(f"{fps = }")
    print(f"video size: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}w {cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}h")

    st.session_state.imageVectors = torch.zeros((frame_count, 512), device=device)
    for i in range(frame_count):
        ret, frame = cap.read()
        with torch.no_grad():
            st.session_state.imageVectors[i] = st.session_state.model.encode_image(
                st.session_state.preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
            )
    st.session_state.imageVectors = torch.mean(st.session_state.imageVectors, axis = 0)
    st.session_state.numpy_tensor = st.session_state.imageVectors.cpu().numpy().tobytes()

def insert_db(file_name, selected_topic, path_location):
    cursor.execute('''
                    INSERT INTO text_vector (file_name, topic, file_location, vectors) values 
                (?, ?, ?, ?)''', (file_name, selected_topic, str(path_location), st.session_state.numpy_tensor))
    conn.commit()
    
def fetch_db(topic):
    cursor.execute("SELECT file_name, topic, file_location, vectors FROM text_vector WHERE topic = ?", (topic,))
    return cursor.fetchall()    

def main():
    Initialization()
    col1, col2 = st.columns([0.7, 0.3])
    with st.sidebar:
        videoInput = st.file_uploader('Enter the video files', type=['mp4'])
        submitBtn = st.button('Submit')
        searchQuery = st.text_input("Enter the text")
        searchBtn = st.button("Search")
    with col1:
        # my_grid = grid([0.5, 0.5], vertical_align="bottom")
        if submitBtn:
            st.write("file uploaded successfully")
            process_video(videoInput)
            similarity_list = []
            for category in categories:
                similarity = similarity_score(category)
                similarity_list.append(similarity)
            selected_category = categories[similarity_list.index(max(similarity_list))]
            st.write(selected_category, max(similarity_list))
            path = f"./{selected_category}"
            if not os.path.exists(path):
                os.mkdir(path)
            destination_path = Path(path, videoInput.name)
            shutil.move(st.session_state.input_file_path, destination_path)
            st.write(str(destination_path))
            insert_db(videoInput.name, selected_category, destination_path)
            
    with col2:
        if searchBtn:
            # st.video(videoInput)
            similarity_list = []
            for category in categories:
                similarities = similarity_score_query(category, searchQuery)
                similarity_list.append(similarities)
            query_selected_category = categories[similarity_list.index(max(similarity_list))]
            st.write(f"Belongs to {query_selected_category} category: {max(similarity_list)}")
            rows = fetch_db(query_selected_category)
            db_data = {}
            for joins in rows:
                file_name, category, file_location, vectors = joins
                vectors = torch.from_numpy(np.frombuffer(vectors, dtype=np.float32))
                st.session_state.imageVectors = vectors
                score = similarity_score(searchQuery)
                db_data.update({score : {"file_name" : file_name,
                                "category" : category,
                                "file_location" : file_location,
                                }})
            video_path = db_data[score]['file_location']
            st.video(video_path)


if __name__ == '__main__':
    main()