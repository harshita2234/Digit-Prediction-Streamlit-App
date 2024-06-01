# Digit Prediction Streamlit Web App

A Streamlit application for predicting handwritten digits using a trained machine learning model.

## Project Structure

- `app.py`: Streamlit app script.
- `Training-pipeline.ipynb`: Jupyter notebook for training the digit prediction model.
- `best_weights.pt`: Trained model weights.
- `utils.py`: Utility functions.
- `models.py`: Model definitions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/harshita2234/Digit-Prediction-Streamlit-App.git
   cd Digit-Prediction-Streamlit-App

2. Install Required Packages:
   ```bash
   python3 -m pip install opencv-python torch numpy streamlit streamlit-drawable-canvas plotly torchvision pandas matplotlib tqdm
   #or
   python -m pip install opencv-python torch numpy streamlit streamlit-drawable-canvas plotly torchvision pandas matplotlib tqdm
3. Unzip the MNIST folder

## Run the App
To run the Streamlit app, navigate to the project directory and execute:
```bash
   streamlit run app.py

