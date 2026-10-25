# Human Activity Recognition

A machine learning project for recognizing human activities from pose/landmark data, with a trained model and live prediction functionality.

## Features
- Activity classification using a trained machine learning model
- Model training using `train_model.py`
- Live activity prediction using `live_predict.py`
- Pose testing utilities

## Project Structure
| File | Description |
|---|---|
| `train_model.py` | Trains the activity recognition model |
| `live_predict.py` | Runs live activity predictions |
| `pose_test.py` | Tests pose-related functionality |
| `font_test.py` | Supporting test script |
| `activity_data.csv` | Activity dataset |
| `activity_model.pkl` | Saved trained model |

## Tech Stack
- Python
- Machine Learning
- Pose/Landmark Data

## How to Run

1. Install the required Python libraries.
2. Run `train_model.py` to train the model.
3. Run `live_predict.py` to test live predictions.

*Note: Install the dependencies required by the scripts before running them.*

## Model Performance

Accuracy and other evaluation metrics will be added after evaluating the model on held-out test data.
