import os
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import logging
import asyncio
import grpc

# Set your Google API Key securely
os.environ["GOOGLE_API_KEY"] = "AIzaSyDpjmKOan8ppAQCJESNTnYBycJGQV26q8o"

# Initialize the ChatGoogleGenerativeAI instance with the specified model
llm = ChatGoogleGenerativeAI(model="gemini-pro")

def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""

def split_text_into_chunks(text, chunk_size=4000, overlap=500):
    """Splits the text into chunks with some overlap to maintain context."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

async def ask_question_on_chunk(pdf_chunk, question, retries=5, backoff_in_seconds=2):
    for attempt in range(retries):
        try:
            combined_query = f"{pdf_chunk}\n\nQuestion: {question}"
            result = await asyncio.to_thread(llm.invoke, combined_query)
            return result.content
        except grpc.RpcError as e:
            logging.warning(f"Attempt {attempt + 1} failed with gRPC error: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(backoff_in_seconds * (2 ** attempt))
            else:
                logging.error("All retry attempts failed.")
                raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

async def ask_question(pdf_text, question):
    chunks = split_text_into_chunks(pdf_text)
    answers = []
    for chunk in chunks:
        answer = await ask_question_on_chunk(chunk, question)
        answers.append(answer)
    # Combine and summarize answers (you can use a more sophisticated method here)
    combined_answer = " ".join(answers)
    return combined_answer

def process_question(pdf_text, question):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    answer = loop.run_until_complete(ask_question(pdf_text, question))
    loop.close()
    return answer

# Streamlit app
st.title("Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = ""

    question = st.text_input("Ask a question about the PDF")

    if st.button("Submit Question"):
        with st.spinner("Processing..."):
            try:
                answer = process_question(pdf_text, question)
                st.session_state.conversation_history += f"Question: {question}\nAnswer: {answer}\n\n"
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.text_area("Conversation History", st.session_state.conversation_history, height=300)
else:
    st.warning("Please upload a PDF file.")