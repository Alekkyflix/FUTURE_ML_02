# Support Ticket Classification System

This project implements a Machine Learning system to classify customer support tickets into categories and assign priority levels. The goal is to help businesses automate ticket triage, reducing response times and improving efficiency.

## Overview

The system processes support tickets (Subject + Description) and predicts:

1.  **Ticket Category**: E.g., Billing, Technical Issue, Product Inquiry.
2.  **Ticket Priority**: E.g., High, Medium, Low, Critical.

## Features

- **Automated Classification**: Uses TF-IDF and Logistic Regression for accurate categorization.
- **Priority Assignment**: Predicts priority based on ticket content using a Random Forest model.
- **Text Preprocessing**: Handles noise, stop words, and lemmatization for better model performance.
- **Easy to Use**: Simple Python script for training and evaluation.

## Methodology

- **Data Source**: `customer_support_tickets.csv`
- **Preprocessing**: Text cleaning (lowercase, remove non-alpha, lemmatization), removal of entries with missing target labels.
- **Models**:
  1.  **Category Classifier**: TF-IDF Vectorizer + Logistic Regression (balanced class weights).
  2.  **Priority Classifier**: TF-IDF Vectorizer + Random Forest Classifier (balanced class weights).

## Installation

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy scikit-learn nltk joblib
    ```
3.  Ensure you have the dataset `customer_support_tickets.csv` in the root directory.

## Usage

### Running the Notebook

1.  Open `Support_ticket_system.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2.  Run all cells to:
    - Load and preprocess the data.
    - Train the Category and Priority classification models.
    - View evaluation metrics and confusion matrices.
    - Save the trained models to the `models/` directory.

### Using the Model (Example)

You can use the saved models or the notebook's `predict_ticket_info` function to classify new tickets.

## Model Evaluation Results

### 1. Category Classification

_Refer to the notebook output for the detailed Classification Report and Confusion Matrix._

- **Key Metrics**: Precision, Recall, and F1-Score are generated for each ticket category.
- **Observations**: The Logistic Regression model effectively separates distinct categories. Common confusion points are visualized in the confusion matrix.

### 2. Priority Classification

_Refer to the notebook output for the detailed Classification Report and Confusion Matrix._

- **Key Metrics**: Precision, Recall, and F1-Score for priority levels (Low, Medium, High, Critical).
- **Observations**: Priority classification is generally harder than topic classification due to the subjective nature of urgency in text. The Random Forest model attempts to capture these nuances.
<img width="919" height="855" alt="image" src="https://github.com/user-attachments/assets/82121545-a3e8-45b1-b28a-6f89acf30839" />

## Structure

- `Support_ticket_system.ipynb`: The main notebook containing the complete pipeline (Data Loading, Cleaning, Training, Evaluation, Visualization).
- `customer_support_tickets.csv`: Dataset used for training.
- `models/`: Directory where trained models are saved.
- `README.md`: Project documentation.
