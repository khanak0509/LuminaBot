import cv2
import os
import base64
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import whisper
from moviepy.editor import VideoFileClip

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

os.environ["GOOGLE_API_KEY"] = "AIzaSyAZfeSc6Db1h-0pBxh24XI_8ZIRtSgL3VM"

model = whisper.load_model("base")

load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyAZfeSc6Db1h-0pBxh24XI_8ZIRtSgL3VM"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    timeout=None,
    max_retries=2,
)

def save_in_txt(text):
    with open('dataset.txt','a') as f:
        f.write(text)
        f.write('\n\n\n')
        print('saved')


def all_video(folder_path):
    """
    Prints the full paths of all image files found in the specified folder
    and its subfolders.
    """
    video_extensions= ('.mp4',)
    video_path = []

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith(video_extensions):
                image_path = os.path.join(dirpath, filename)
                video_path.append(image_path)
    return(video_path)

def img_captioning(image_path):
    image = cv2.imread(image_path)
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _, buffer = cv2.imencode('.png', rgb_frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    """You are an AI assistant generating captions for images that will be used in a semantic search system. 

    Instructions:
    1. Write a **short, clear, and factual description** of the image (1–3 sentences).
    2. Include **key objects, people, animals, vehicles, signs, or actions** in the scene.
    3. Mention **any obvious colors, positions, or spatial relations** if important.
    4. **Do not add subjective opinions** or speculative information.
    5. Keep it **consistent in style** across multiple images to help embeddings match queries accurately.
    6. Output **plain text** only (no punctuation errors or extra formatting).

    Example:
    - Image of a street with cars and a traffic light → "A busy city street with three cars waiting at a red traffic light. Pedestrians are crossing the road."
    - Image of a cat on a sofa → "A gray cat sitting on a beige sofa looking toward the camera."
    """
                )
            },
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{encoded_image}"
            }
        ]
    )
    response = llm.invoke([message])
    save_in_txt(response.content)



def audio_text(audio_path):
    result = model.transcribe(audio_path)
    save_in_txt(result['text'])


def all_audio(folder_path):
    """
    Prints the full paths of all image files found in the specified folder
    and its subfolders.
    """
    audio_extensions= ('.mp3','.mp4','.wav')
    audio_path = []

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith(audio_extensions):
                image_path = os.path.join(dirpath, filename)
                audio_path.append(image_path)
    return(audio_path)

def all_image(folder_path):
    """
    Prints the full paths of all image files found in the specified folder
    and its subfolders.
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    image_paths = []

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(dirpath, filename)
                image_paths.append(image_path)
    return(image_paths)


                
folder = '.'

def extract_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)
    video_clip.close()



for video_path in all_video(folder):
    extract_audio(video_path,video_path.replace('4','3'))


for image in all_image(folder):
    img_captioning(image)


for audio in all_audio(folder):
    audio_text(audio)



    
