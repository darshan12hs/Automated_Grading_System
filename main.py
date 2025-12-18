import streamlit as st
from PIL import Image
import io
import aspose.ocr as ocr
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from cv import grade_test_image
import numpy as np

UPLOAD_FOLDER = 'uploaded_files'
# Initialize an instance of Aspose.OCR API
api = ocr.AsposeOcr()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Title of the app
st.title("Image Upload and OCR App with BERT Text Similarity")

# Description
st.write("Upload an image to extract text using OCR and compare it with a correct answer using BERT.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to get BERT embedding for a given text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Function to evaluate similarity between extracted and correct answers
def evaluate_answer(extracted, correct):
    extracted_embedding = get_embedding(extracted).numpy()
    correct_embedding = get_embedding(correct).numpy()
    similarity = cosine_similarity([extracted_embedding], [correct_embedding])[0][0]
    return similarity

# Function to score answers
def score_answers_advanced(extracted_answers, correct_answers, max_score,score2):
    global total
    total_similarity = 0
    for extracted, correct in zip(extracted_answers, correct_answers):
        similarity = evaluate_answer(extracted, correct)
        total_similarity += similarity
        total=(total_similarity / len(correct_answers)) * max_score
    return (total + score2) / 2.0

def image_bytes(u_file):
    # to read as byte code
    byte_data = u_file.getvalue()
    # to open the file
    image = Image.open(io.BytesIO(byte_data))
    return image,byte_data

# Check if a file has been uploaded
if uploaded_file is not None:
    im,by = image_bytes(uploaded_file)
    #converting from bytes to numpy array
    img_array = np.array(im)
    score,AnsImg = grade_test_image(img_array)

    # convert from numpy array to pil image
    PilImage = Image.fromarray(AnsImg)
    byte_io = io.BytesIO()
    PilImage.save(byte_io, format='JPEG')
    #convert from Pil image to bytes
    bytes_data = byte_io.getvalue()
    image = Image.open(io.BytesIO(bytes_data))


    # Display image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Save the uploaded image temporarily to disk
    with open("temp_image.png", "wb") as f:
        f.write(bytes_data)

    # Add image to the recognition batch
    input = ocr.OcrInput(ocr.InputType.SINGLE_IMAGE)
    input.add("temp_image.png")

    # Extract and show text
    results = api.recognize_handwritten_text(input)
    extracted_text = results[0].recognition_text
    st.write("Recognized Text:")
    st.write(extracted_text)

    # Input correct answer for comparison
    correct_answer = st.text_input("Enter the correct answer text:")

    if correct_answer:
        # Score the extracted text against the correct answer
        extracted_answers = [extracted_text]
        correct_answers = [correct_answer]
        score2, img = grade_test_image(img_array)
        score1 = score_answers_advanced(extracted_answers, correct_answers, 100,score2)
        st.write(f"Total Score: {score1:.2f}")
