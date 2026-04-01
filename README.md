# Automated Grading System

I built this project to automate the grading process for both multiple-choice and written exam responses by combining computer vision with natural language processing.

## What it does

The system takes scanned answer sheets as input and processes them in two stages:

- First, it uses OpenCV to detect and interpret marked bubbles for objective questions. These are matched against a predefined answer key to generate scores automatically.  
- For written responses, a BERT-based model evaluates answers by comparing their meaning with reference solutions. Instead of simple keyword matching, it focuses on semantic similarity to produce more accurate scoring.

## Why this project

Grading exams manually can be time-consuming and inconsistent, especially when subjective answers are involved. This project explores how machine learning can make the process faster, more scalable, and more consistent while still capturing the quality of written responses.

## Main Components

- **cv.py** – Handles image processing, detects answer regions, and extracts marked responses from scanned sheets  
- **main.py** – Integrates all components and runs the full grading workflow  

## Tech Stack

- Python (3.8+)  
- OpenCV  
- Hugging Face Transformers (BERT)  
- NumPy  
- Pandas  

## How to Run

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/automated-grading-system.git
cd automated-grading-system
pip install -r requirements.txt

Run the project:
python main.py
