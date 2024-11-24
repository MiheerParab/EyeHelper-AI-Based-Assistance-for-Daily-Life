import streamlit as st
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
import pytesseract
from gtts import gTTS
import io
import base64
import logging

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the Streamlit page configuration
st.set_page_config(page_title="EyeHelper - AI Companion for the Visually Impaired", layout="wide")

# Load API credentials from environment variables or configuration file
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()  # Load environment variables from the .env file
    GOOGLE_API_KEY = "AIzaSyCU4o0dyuHFZV9Bsvmg9LwxNnXEO8An6Lg"  # Secure API Key access
except ImportError:
    logging.error("Ensure `python-dotenv` is installed and your .env file is correctly configured.")

# Initialize LangChain models for language and vision tasks
language_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
vision_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Function to handle and log errors gracefully
def log_error(exception):
    logging.error(f"An error occurred: {exception}")
    st.error(f"Something went wrong: {exception}")

# Image-based Scene Analysis: Generate descriptions based on visual data
def analyze_scene(image):
    try:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        encoded_image = base64.b64encode(image_bytes.getvalue()).decode()

        prompt = {
            "type": "text",
            "text": """Provide a detailed description of this image for visually impaired users:
            - Describe the scene, including key objects and their locations
            - Include any people, their actions, and the environment's lighting
            - Identify colors and notable features in the scene."""
        }
        image_data = {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{encoded_image}"
        }

        response = vision_model.invoke([HumanMessage(content=[prompt, image_data])])
        return response.content
    except Exception as e:
        log_error(e)

# Extracting Text from Image using OCR: Process and structure text for accessibility
def extract_text(image):
    try:
        text = pytesseract.image_to_string(image).strip()
        if not text:
            return "No text found in the image."

        # Create a structured prompt for text enhancement
        template = PromptTemplate(
            input_variables=["text"],
            template="""Enhance and structure the following extracted text for a visually impaired individual:
            TEXT: {text}
            Please:
            1. Correct any OCR mistakes
            2. Format the content clearly
            3. Highlight significant details like dates or numbers"""
        )

        enhanced_text = LLMChain(llm=language_model, prompt=template).run(text=text)
        return enhanced_text
    except Exception as e:
        log_error(e)

# Detect objects and assess safety: Identify obstacles and provide navigation suggestions
def analyze_objects(image):
    try:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        encoded_image = base64.b64encode(image_bytes.getvalue()).decode()

        prompt = {
            "type": "text",
            "text": """Analyze the image for safety and navigation:
            - Identify potential obstacles or hazards
            - Suggest the safest paths
            - Provide distance estimates and spatial relationships"""
        }
        image_data = {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{encoded_image}"
        }

        response = vision_model.invoke([HumanMessage(content=[prompt, image_data])])
        return response.content
    except Exception as e:
        log_error(e)

# Provide assistance with specific tasks based on the image
def assist_with_tasks(image, task):
    try:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        encoded_image = base64.b64encode(image_bytes.getvalue()).decode()

        task_prompts = {
            "item_identification": "Identify items in the image and describe them.",
            "label_reading": "Analyze and read any visible labels in the image.",
            "navigation_help": "Provide advice on safe navigation within this environment.",
            "daily_tasks": "Assist with common daily tasks such as finding objects or reading text."
        }

        prompt = {
            "type": "text",
            "text": task_prompts.get(task, task_prompts["item_identification"])
        }
        image_data = {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{encoded_image}"
        }

        response = vision_model.invoke([HumanMessage(content=[prompt, image_data])])
        return response.content
    except Exception as e:
        log_error(e)

# Convert text to speech for easy listening
def convert_text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio = io.BytesIO()
        tts.write_to_fp(audio)
        audio.seek(0)
        return audio.getvalue()
    except Exception as e:
        log_error(e)

# Add custom CSS for styling the Streamlit interface
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;  /* Light blue background */
            font-family: 'Helvetica', sans-serif;
            color: #333;
        }
        .stButton button {
            background-color: #008CBA;  /* Sky blue */
            color: white;
            padding: 14px 28px;
            font-size: 20px;
            font-weight: bold;
            border-radius: 10px;
            border: 2px solid #006f8e;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #005f75;  /* Darker blue */
            transform: scale(1.05);  /* Slight scale effect */
        }
        .stButton button:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.5);  /* Outline for focus state */
        }
        .stRadio div, .stSelectbox div {
            font-size: 18px;
            color: #333;
        }
        .stText {
            font-size: 18px;
            color: #555;
        }
        .stSidebar {
            background-color: #E0F7FA;  /* Light cyan */
            padding: 20px;
        }
        .stMarkdown {
            font-size: 18px;
            color: #333;
        }
        .stImage img {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .footer {
            background-color: #f0f8ff;
            padding: 20px;
            font-size: 14px;
            color: #495057;
            text-align: center;
        }
        .features-container {
            display: flex;
            justify-content: center;
            flex-direction: column;
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)

# Main function to control the app's flow
def main():
    st.title("EyeHelper: AI-Based Assistance for Daily Life")

    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    # Feature Buttons with custom styling
    st.markdown('<div class="features-container">', unsafe_allow_html=True)
    selected_feature = st.radio(
        "Select a Feature",
        ["Scene Description", "Text Extraction", "Object Detection", "Task Assistance"],
        help="Choose the feature that best suits your needs."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if selected_feature == "Scene Description":
            if st.button("Analyze Scene", use_container_width=True):
                with st.spinner("Processing scene description..."):
                    scene_description = analyze_scene(image)
                    st.write(scene_description)
                    audio_output = convert_text_to_speech(scene_description)
                    st.audio(audio_output, format="audio/mp3")

        elif selected_feature == "Text Extraction":
            if st.button("Extract and Read Text", use_container_width=True):
                with st.spinner("Extracting text from the image..."):
                    text = extract_text(image)
                    st.write(text)
                    audio_output = convert_text_to_speech(text)
                    st.audio(audio_output, format="audio/mp3")

        elif selected_feature == "Object Detection":
            if st.button("Analyze Objects", use_container_width=True):
                with st.spinner("Detecting objects in the scene..."):
                    object_analysis = analyze_objects(image)
                    st.write(object_analysis)
                    audio_output = convert_text_to_speech(object_analysis)
                    st.audio(audio_output, format="audio/mp3")

        elif selected_feature == "Task Assistance":
            task = st.selectbox("Select a Task", ["item_identification", "label_reading", "navigation_help", "daily_tasks"])
            if st.button("Assist with Task", use_container_width=True):
                with st.spinner("Providing assistance..."):
                    task_assistance_result = assist_with_tasks(image, task)
                    st.write(task_assistance_result)
                    audio_output = convert_text_to_speech(task_assistance_result)
                    st.audio(audio_output, format="audio/mp3")

    # About Section in Footer
    st.markdown("""
        <div class="footer">
            <strong>About EyeHelper:</strong><br>
            EyeHelper is an AI-powered companion designed to assist visually impaired individuals in navigating the world around them.
            By analyzing images and providing real-time feedback through text and voice, EyeHelper enhances independence and accessibility for users,
            enabling them to perform everyday tasks with ease.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
