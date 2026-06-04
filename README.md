# AI Data Classification Using KNN

## Project Overview

This project is a supervised machine learning classification system developed as part of the DecodeLabs Artificial Intelligence Internship Project 2.

The system uses the Iris dataset and applies the K-Nearest Neighbors algorithm to classify Iris flowers into three species:

* Setosa
* Versicolor
* Virginica

The project includes data preprocessing, feature scaling, model training, model evaluation, and a Flask-based web interface for real-time prediction.

## Project Objective

The main objective of this project is to build a basic AI classification model using a small dataset and demonstrate the complete supervised learning workflow.

## Features

* Load and analyze the Iris dataset
* Split data into training and testing sets
* Apply feature scaling using StandardScaler
* Train a K-Nearest Neighbors classifier
* Evaluate the model using accuracy, confusion matrix, precision, recall, and F1 score
* Save the trained model using Joblib
* Provide a Flask web interface for predictions
* Display prediction results clearly

## Technologies Used

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* HTML
* CSS

## Machine Learning Algorithm

### K-Nearest Neighbors

K-Nearest Neighbors is a supervised machine learning algorithm used for classification. It classifies a new data point by checking the nearest data points in the training dataset and selecting the majority class.

In this project, K = 5 is used.

## Dataset

The Iris dataset contains 150 flower samples with 4 features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Target classes:

* Setosa
* Versicolor
* Virginica

## Project Structure

```text
AI-Data-Classification-Using-KNN/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   └── iris_knn_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   └── model_report.txt
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
└── screenshots/
```

## How to Run

Clone the repository, install dependencies, train the model, and run the Flask app.

```bash
git clone https://github.com/MAHESHISAGARIKA/Task-2-MaheshiRajapaksha.git
cd AI-Data-Classification-Using-KNN

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python train_model.py
python app.py
```

Open the application in the browser:

```text
http://127.0.0.1:5000
```

