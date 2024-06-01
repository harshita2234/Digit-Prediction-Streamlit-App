# Digit Prediction Streamlit Web App

A Streamlit application for predicting handwritten digits using a trained machine learning model.
<img width="490" alt="image" src="https://github.com/harshita2234/Digit-Prediction-Streamlit-App/assets/97393648/c775dd0d-eb9e-46b8-b37b-d7109325afb8">

<img width="485" alt="image" src="https://github.com/harshita2234/Digit-Prediction-Streamlit-App/assets/97393648/8d4e77db-0875-4447-9eca-287bf30ef8c2">
<img width="564" alt="image" src="https://github.com/harshita2234/Digit-Prediction-Streamlit-App/assets/97393648/5fe41129-9b15-4af9-aaba-194719700dcb">



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

