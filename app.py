import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# Model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# File loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts

# LLM pipeline
def llm_pipeline(input_text):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

# Function to display the PDF of a given file
@st.cache_data
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code
st.set_page_config(layout="wide")

def main():
    st.title("STUDY GUIDE")

    #st.sidebar.header("Options")
    summarization_mode = st.sidebar.radio("LEARNING MODE :", ["PDF Summarization", "Text Summarization", "Answer for Exam",  "Mind Map", "Understand Like a 10-Year-Old", "Flashcards", "Mneumonic", "Video Explanation", "Audio Explanation"])
    
    if summarization_mode == "PDF Summarization":
        uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
        if uploaded_file is not None:
            if st.button("Summarize"):
                col1, col2 = st.columns(2)
                filepath = "data/"+uploaded_file.name
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                with col1:
                    st.info("Uploaded File")
                    pdf_view = displayPDF(filepath)

                with col2:
                    summary = llm_pipeline(file_preprocessing(filepath))
                    st.info("Summarization Complete")
                    st.success(summary)
    
    if summarization_mode == "Text Summarization":
        text_input = st.text_area("Enter the text you want to summarize:", height=200)
        if st.button("Summarize"):
            if text_input:
                summary = llm_pipeline(text_input)
                st.info("Summarization Complete")
                st.success(summary)
            else:
                st.warning("Please enter some text to summarize.")
                
    # if summarization_mode == "Answer for Exam":
         # if exam_type=="MCQ":
             # mcq_question = st.selectbox("", ["Question 1","Question 2"])
             # if mcq_question=='Question 1':
                # optionA = st.radio('', ['Option A', 'Option B'], index=None,)
                # if (optionA==['Option A']):
                    # answer = "The first question has two options and only one of them is correct."
                #elif (optionA==['Option B']):
                    
                    

if __name__ == "__main__":
    main()
