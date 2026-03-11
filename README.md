# Aircraft-Performance-and-Engine-Life-prediction-using-Machine-Learning
# Turbofan Engine Predictive Maintenance (RUL Prediction)

This project predicts the **Remaining Useful Life (RUL)** of turbofan engines using sensor data from the **NASA CMAPSS dataset**. The objective is to analyze engine degradation patterns and estimate how many operational cycles remain before engine failure using machine learning.

---

## Dataset

This project uses the NASA Turbofan Engine Degradation Simulation dataset.

:contentReference[oaicite:0]{index=0}

The dataset contains multiple engines operating over time until failure with recorded sensor measurements.

### Files

| File | Description |
|-----|-------------|
| `PM_train` | Training data with full engine life cycles |
| `PM_test` | Test data where engines stop before failure |
| `PM_truth` | Actual Remaining Useful Life values for test engines |

---

## Dataset Structure

Each row represents one **engine cycle**.

| Column | Description |
|------|-------------|
| id | Engine unit number |
| cycle | Operating cycle |
| setting1-3 | Engine operational settings |
| s1–s21 | Sensor measurements |

---

## Remaining Useful Life (RUL)

RUL is calculated as: RUL = max_cycle - current_cycle

The model learns to predict how many cycles remain before engine failure.

---

## Project Workflow

1. Data Loading  
2. Data Preprocessing  
3. RUL Calculation  
4. Exploratory Data Analysis  
5. Feature Engineering  
6. Model Training  
7. Model Evaluation

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  
- Scikit-learn  
- TensorFlow / Keras  
- Google Colab  

---

