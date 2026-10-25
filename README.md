# Human Activity Recognition

A machine learning project that classifies human activities from pose landmark data using a Random Forest classifier, with support for live prediction.

## Model Performance

- **Classifier:** Random Forest
- **Number of estimators:** 100
- **Training split:** 80%
- **Test split:** 20%
- **Test accuracy:** 96.48%
- **Split configuration:** `random_state=42`

Accuracy is measured on the held-out test split. Further evaluation is needed to assess generalization to unseen people and recordings.

## Features

- Train a Random Forest activity classifier
- Predict activity labels from pose landmark features
- Save and load the trained model
- Perform live activity prediction using a webcam

## Project Structure

| File | Description |
|---|---|
| `train_model.py` | Trains and evaluates the classifier |
| `live_predict.py` | Performs live activity prediction |
| `pose_test.py` | Pose testing utility |
| `font_test.py` | Supporting test script |
| `activity_data.csv` | Activity dataset |
| `activity_model.pkl` | Saved trained model |

## Tech Stack

- Python
- Pandas
- Scikit-learn
- OpenCV
- MediaPipe
- Joblib

## How to Run

Install the required libraries, then run:

```bash
python train_model.py
python live_predict.py
