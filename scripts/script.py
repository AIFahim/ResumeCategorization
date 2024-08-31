import os
import sys
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import shutil
from PyPDF2 import PdfReader
import numpy as np
import gdown
import zipfile

def download_and_extract_zip_from_gdrive(file_id, extract_to):
    """
    Download a zip file from Google Drive and extract it to a specified directory.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "/tmp/model.zip"
    gdown.download(url, output, quiet=False)

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def load_model_and_tokenizer(model_dir):
    """
    Load the fine-tuned model and tokenizer from the specified directory.
    """
    # Check if "checkpoint-1744" is already in the path
    if "checkpoint-1744" not in model_dir:
        model_dir = os.path.join(model_dir, "checkpoint-1744")

    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # Load the saved label encoder classes
    label_classes = np.load(os.path.join(model_dir, "label_classes.npy"), allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_classes

    return model, tokenizer, label_encoder


def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.
    """
    pdfreader = PdfReader(pdf_path)
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def categorize_resumes(directory, model, tokenizer, label_encoder):
    """
    Categorize resumes in the specified directory and move them to respective category folders.
    """
    categorized_resumes = []

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if not os.path.isfile(file_path) or not filename.lower().endswith('.pdf'):
            continue

        # Extract text from the PDF file
        resume_text = extract_text_from_pdf(file_path)

        # Tokenize the text
        inputs = tokenizer.encode_plus(
            resume_text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()

        # Get the category name from label encoder
        category = label_encoder.inverse_transform([predicted_label])[0]

        # Create the category directory if it doesn't exist
        category_dir = os.path.join(directory, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

        # Move the file to the respective category folder
        shutil.move(file_path, os.path.join(category_dir, filename))

        # Store the filename and category for the CSV file
        categorized_resumes.append({'filename': filename, 'category': category})

    return categorized_resumes

def save_categorized_resumes(categorized_resumes, output_csv='categorized_resumes.csv'):
    """
    Save the categorized resumes information to a CSV file.
    """
    df = pd.DataFrame(categorized_resumes)
    df.to_csv(output_csv, index=False)
    print(f"Categorized resumes saved to {output_csv}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py path/to/dir")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    # Google Drive file ID
    file_id = "1oQD8tPI1svFcR9AjSW6ANRoXYaTLHDuJ"  # Replace with your actual Google Drive file ID

    # Model directory path
    model_dir = "./model_dir"

    # Download and extract the zip file to the model directory
    download_and_extract_zip_from_gdrive(file_id, model_dir)

    # Load the model, tokenizer, and label encoder
    model, tokenizer, label_encoder = load_model_and_tokenizer(model_dir)

    # Categorize resumes and save the categorized resumes information
    categorized_resumes = categorize_resumes(directory, model, tokenizer, label_encoder)
    save_categorized_resumes(categorized_resumes)

if __name__ == "__main__":
    main()