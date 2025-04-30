# 🧠 Probabilistic Forecasting of Walmart Sales using LSTM (M5 Competition)

This project applies deep learning for probabilistic forecasting on Walmart retail sales data, specifically targeting the M5 Uncertainty Forecasting Challenge hosted on Kaggle. It uses a multivariate LSTM model with customized quantile loss functions to generate accurate prediction intervals across hierarchical time series.

## 📊 Problem Context

The M5 competition consists of two tasks:
- **Accuracy Competition**: Predict point estimates
- **Uncertainty Competition**: Predict quantiles for probabilistic forecasting

This project focuses on the latter, forecasting sales for the **Hobbies** category across **store and department levels** (Level 9 aggregation).

## 🧪 Techniques Used

- Long Short-Term Memory (LSTM) Neural Network
- Multivariate time series inputs: past sales, weekday, events, SNAP days
- Custom **quantile loss functions** for probabilistic predictions
- Evaluation using **Weighted Scaled Pinball Loss (WSPL)**
- Benchmark: compared against ARIMA bottom-up approach

## 🧰 Tech Stack

- Python 3.8+
- pandas, numpy, scikit-learn
- TensorFlow / Keras
- Custom loss functions for quantile regression

## 📁 Files

| File                              | Description                                     |
|-----------------------------------|-------------------------------------------------|
| `predict_storedept_all_features.py` | Core model training and prediction script       |
| `calendar.csv`                   | Calendar with holidays, events, SNAP, etc.     |
| `sales_train_validation.csv`     | Walmart training sales data                    |
| `sales_test_validation.csv`      | Walmart evaluation data                        |
| `Script_Presentation_Sem2.docx`  | Accompanying script from oral presentation     |
| `Capstone_Presentation_Sem2.pptx`| Final capstone presentation slides             |
| `docs/`                          | M5 competition background and results papers   |

## 🧠 Learnings

- Built custom loss functions to forecast 9 quantiles
- Modeled multivariate inputs in LSTM for time series
- Conducted feature importance with Random Forest
- Achieved **62% improvement over ARIMA** on validation set

## 🎓 Academic Info

This was the capstone project for **DATA7901 – Data Science Capstone** at the University of Queensland, submitted as part of the **Master's in Data Science**.

## 🚀 How to Run

1. Install required packages:
   ```bash
   pip install -r requirements.txt
