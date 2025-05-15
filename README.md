# FiT-AI Trainer

This project uses a Temporal Convolutional Network (TCN) to classify workout movements as **correct** or **incorrect type** by analyzing human pose data extracted from videos using MediaPipe.

However, this current version can only detect squat workouts and only classify whether the form is correct or incorrect based on pose landmark sequences. It does not yet support multiple workout types or detailed mistake classification, but serves as a prototype for FiT-AI Traniner Application.

---

## Model Training and API Integration Steps

### 1. Clone the repository
```
git clone https://github.com/Pichayanon/FiT-AI.git
```
### 2. Navigate to the backend 
```
cd backend
```
### 3. Set up a virtual environment
- mac/linux
```
python3 -m venv venv
source venv/bin/activate  
```
- window
```
python3 -m venv venv
venv\Scripts\activate
```
### 4. Install Python dependencies
```
pip install -r requirements.txt
```
### 5. Extract Squat Sequences from videos
```
python extract_squat_from_video.py
```
This script will read .mp4 videos from the video/ folder and generate a ```squat_data.csv``` file containing labeled sequences.
### 6. Train the TCN model
```
python train_tcn_model_squat.py
```
This will create or update the model file: ```squat_tcn_model.h5```

### 7. Run the model RestfulAPI
```
python squat_model_api.py
```
This starts a Flask-based REST API server on http://localhost:5050.

Once the server is running:

- You can send a POST request to the /predict endpoint with a 30×48 pose sequence in JSON format.

- The API loads the trained squat_tcn_model.h5, processes the input sequence, and returns:

  - A prediction (0 = incorrect, 1 = correct)

  - A confidence score (between 0.0 and 1.0)
---

## Running iOS Application steps
Open the .xcworkspace or .xcodeproj file using Xcode, and:

- Select a simulator device (e.g., iPhone 16)

- Click Run to launch the app
