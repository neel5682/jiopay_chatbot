import json
import numpy as np
import faiss
import google.generativeai as genai
import streamlit as st
import time
import logging
from functools import lru_cache
from langchain.text_splitter import RecursiveCharacterTextSplitter
import speech_recognition as sr
from gtts import gTTS
import os

# Configure Gemini API
genai.configure(api_key="AIzaSyCuvt31hJeflv7Hxlg9UOBZdlxEbOJ-2JM")  # Replace with your Gemini API key

# Load the dataset
with open("jiopay_faqs.json", "r") as file:
    data = json.load(file)

# Flatten the dataset into a list of question-answer pairs
faq_pairs = []
for category, questions in data.items():
    for question, answer in questions.items():
        faq_pairs.append({
            "question": question,
            "answer": answer,
            "category": category
        })

print(f"Total FAQs: {len(faq_pairs)}")

# Function to generate embeddings
def get_embedding(text):
    model = genai.embed_content(
        model="models/embedding-001",  # Use the correct embedding model
        content=text,
        task_type="retrieval_query"
    )
    return model["embedding"]

# Generate embeddings for all FAQs
for faq in faq_pairs:
    faq["embedding"] = get_embedding(faq["question"])

print("Embeddings generated successfully!")

# Build a FAISS index
embeddings = np.array([faq["embedding"] for faq in faq_pairs])
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
index.add(embeddings)

print("FAISS index created successfully!")

# Function to retrieve relevant FAQs
def retrieve_relevant_faqs(query, top_k=3):
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding])
    distances, indices = index.search(query_embedding, top_k)
    relevant_faqs = [faq_pairs[i] for i in indices[0]]
    return relevant_faqs, distances[0]  # Return distances for relevance check

# Function to check if retrieved FAQs are relevant
def is_relevant(distances, threshold=0.5):
    # If the smallest distance is below the threshold, the FAQs are relevant
    return any(distance < threshold for distance in distances)

# Function to generate a response
def generate_response(query, context):
    context_str = "\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in context])
    model = genai.GenerativeModel("models/gemini-pro")  # Use the correct model name
    response = model.generate_content(
        f"Based on the following context, answer the user's question concisely:\n{context_str}\n\nQuestion: {query}"
    )
    return response.text

# Function to handle greetings
GREETINGS = ["hi", "hello", "good morning", "good afternoon", "good evening"]

def handle_greeting(query):
    if query.lower() in GREETINGS:
        return "Hello! Welcome to JioPay Customer Support. How can I assist you today?"
    return None

# Function to handle out-of-scope queries
def handle_out_of_scope(query):
    return "I'm sorry, I couldn't find an answer to your question. Please contact JioPay support for further assistance."

# Function to handle empty queries
def handle_empty_query(query):
    if not query.strip():
        return "Please enter a question or concern."
    return None

# Function to display a typing indicator
def show_typing_indicator():
    with st.empty():
        for _ in range(3):
            st.write("Typing...")
            time.sleep(0.5)
            st.write("")
            time.sleep(0.5)

# Function to display sources clearly
def display_sources(relevant_faqs):
    st.write("**Sources:**")
    for faq in relevant_faqs:
        st.write(f"- **{faq['question']}** (Category: {faq['category']})")

# Function to add a feedback mechanism
def add_feedback():
    feedback = st.radio("Was this response helpful?", ("Yes", "No"))
    if feedback:
        st.write(f"Thank you for your feedback! You selected: {feedback}")

# Function to log interactions
logging.basicConfig(filename="chatbot.log", level=logging.INFO)

def log_interaction(user_id, query, response):
    logging.info(f"User: {user_id}, Query: {query}, Response: {response}")

# Function to handle voice input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.error("Sorry, there was an issue with the speech recognition service.")
            return None

# Function to handle text-to-speech output
def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    os.system("start response.mp3")  # Play the audio (Windows)
    # For macOS/Linux, use: os.system("afplay response.mp3")

# Streamlit app
st.title("JioPay Customer Support Chatbot")
st.write("Welcome to JioPay Customer Support! How can I assist you today?")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Voice input button
if st.button("🎤 Use Voice Input"):
    user_query = get_voice_input()
else:
    user_query = st.chat_input("Ask a question:")

if user_query:
    # Handle empty queries
    if not user_query.strip():
        st.warning("Please enter a question or concern.")
    else:
        # Handle greetings
        greeting_response = handle_greeting(user_query)
        if greeting_response:
            st.session_state.messages.append({"role": "assistant", "content": greeting_response})
            with st.chat_message("assistant"):
                st.markdown(greeting_response)
        else:
            # Show typing indicator
            show_typing_indicator()

            # Retrieve relevant FAQs and their distances
            relevant_faqs, distances = retrieve_relevant_faqs(user_query)

            # Check if the retrieved FAQs are relevant
            if is_relevant(distances):
                # Generate a response based on the retrieved FAQs
                response = generate_response(user_query, relevant_faqs)
            else:
                # Use a fallback response for out-of-scope queries
                response = handle_out_of_scope(user_query)

            # Display the response
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

            # Display source citations (if FAQs are relevant)
            if is_relevant(distances):
                display_sources(relevant_faqs)

            # Add feedback mechanism
            add_feedback()

            # Log the interaction
            log_interaction("user_123", user_query, response)

            # Text-to-speech output
            text_to_speech(response)

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()