# NBA 10-Day Contract Predictor

## Project Overview
This project builds machine learning models to predict whether an NBA player on a 10-day contract.

## Step 1: Data Loading & Feature Selection
**File:** `load_data.py`

In this step, we isolated our target variable (`SEASON`) and removed data that would negatively impact the model:
* **`Unnamed: 0`:** This is just a row index and holds no mathematical value.
* **`PLAYER`:** a player's name does not effect their chance of getting a contract.
* **`season year`:** A timestamp that won't help our model predict the outcome for players in future years (can only hurt future prediction). 
* **`SEASON`:** Can't use it in training because it has 1:1 correlation with the target.

## Step 2: Data Preprocessing
*(To be filled: Explain One-Hot Encoding and Scaling)*

## Step 3: Classic Machine Learning (Random Forest)
*(To be filled: Explain why we chose Random Forest and its results)*

## Step 4: Neural Network (Keras)
*(To be filled: Explain the architecture and loss function)*