import cv2
import os
import base64
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyAZfeSc6Db1h-0pBxh24XI_8ZIRtSgL3VM"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    timeout=None,
    max_retries=2,
)

image = cv2.imread('image.png')
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
print(response.content)
