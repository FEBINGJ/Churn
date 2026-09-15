# Customer Churn Prediction

This small project demonstrates binary classification without scikit-learn. It generates a reproducible customer dataset, trains logistic regression with NumPy, evaluates predictions, and creates a Seaborn/Matplotlib chart.

## Run

From this folder:

```powershell
C:/Python314/python.exe churn_prediction.py
```

The script prints accuracy, precision, recall, and an example prediction. It saves the chart to `output/churn_analysis.png`.

## Features

- `tenure_months`: how long the customer has been subscribed
- `monthly_charges`: monthly bill
- `support_calls`: recent support calls
- `contract_length`: contract duration in months

The data is synthetic, so the project is for learning and demonstration rather than production decisions.