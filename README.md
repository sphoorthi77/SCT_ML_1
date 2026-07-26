# 🏠 House Price Prediction

A machine learning model that predicts house prices based on key property features — square footage, number of bedrooms, and number of bathrooms — using Linear Regression. The project includes full evaluation metrics, visualizations, and an interactive command-line prediction tool.

## 📌 Project Goal

Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

---

## 📂 Dataset

The dataset (`house_price_regression_dataset.csv`) contains **1,000 house records** with the following columns:

| Column | Description |
|---|---|
| `Square_Footage` | Total livable area of the house |
| `Num_Bedrooms` | Number of bedrooms |
| `Num_Bathrooms` | Number of bathrooms |
| `Year_Built` | Year the house was built |
| `Lot_Size` | Size of the land lot |
| `Garage_Size` | Number of garage spaces |
| `Neighborhood_Quality` | Rating of the neighborhood (1–10) |
| `House_Price` | Actual sale price (target variable) |

For this model, only three features are used for prediction — **Square Footage**, **Number of Bedrooms**, and **Number of Bathrooms** — since these are the most direct, easy-to-provide inputs for a quick price estimate.

---

## ⚙️ How the Model Works

1. **Data loading** — The CSV is loaded with pandas.
2. **Feature selection** — `Square_Footage`, `Num_Bedrooms`, and `Num_Bathrooms` are used as input features; `House_Price` is the prediction target.
3. **Train/test split** — 80% of the data is used for training, 20% is held out for testing.
4. **Model** — A **Linear Regression** model is trained to find the best-fit relationship between the features and house price.
5. **Evaluation** — The model's predictions on the test set are compared against actual prices using multiple metrics (see below).
6. **Visualization** — Two plots are generated and saved to `model_evaluation.png`: an Actual vs Predicted scatter plot, and a confusion matrix for price-category classification.
7. **Interactive prediction** — After training, the script asks for square footage, bedrooms, and bathrooms in the terminal, and prints a predicted price.

---

## 📊 Results

```
Square_Footage coefficient : 201.16
Num_Bedrooms coefficient   : 10,111.38
Num_Bathrooms coefficient  : 9,342.06
Intercept                  : 4,092.03
```

This means, holding other features constant, each additional square foot adds about **$201** to the predicted price, each additional bedroom adds about **$10,111**, and each additional bathroom adds about **$9,342**.

**Regression metrics on the test set:**

| Metric | Value | Meaning |
|---|---|---|
| MAE (Mean Absolute Error) | $23,384.98 | On average, predictions are off by about $23,385 |
| MSE (Mean Squared Error) | 846,377,534.22 | Average squared prediction error |
| RMSE (Root Mean Squared Error) | $29,092.57 | Typical prediction error, in dollars |
| R² (Coefficient of Determination) | 0.9869 | The model explains **98.7%** of the variation in house prices |

**Price category accuracy:** as an extra, easier-to-read view of performance, actual and predicted prices are also grouped into 4 categories (Low / Medium / High / Very High), and compared like a classification problem:

```
Price Category Accuracy: 93.50%
```

Confusion matrix (rows = actual category, columns = predicted category):

| | Low | Medium | High | Very High |
|---|---|---|---|---|
| **Low** | 28 | 0 | 0 | 0 |
| **Medium** | 3 | 63 | 0 | 0 |
| **High** | 0 | 1 | 66 | 4 |
| **Very High** | 0 | 0 | 5 | 30 |

Most of the model's mistakes happen at the boundary between adjacent price categories (e.g. predicting "High" instead of "Very High"), which makes sense for a continuous regression problem being viewed through a categorical lens — the model isn't confusing cheap houses for expensive ones, it's just occasionally landing just above or below a category cutoff.

A saved plot, `model_evaluation.png`, visualizes both the actual-vs-predicted scatter plot and this confusion matrix side by side.

---

## 🧰 Requirements

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

| Library | Purpose |
|---|---|
| `pandas` | Loading and handling the dataset |
| `numpy` | Numerical operations (e.g. RMSE calculation) |
| `matplotlib` | Plotting the evaluation graphs |
| `scikit-learn` | Model training, splitting, and evaluation metrics |
| `joblib` | Saving the trained model to disk |

---

## 🚀 How to Run

1. Make sure `house_price_regression.py` and `house_price_regression_dataset.csv` are in the same folder.
2. Run:

```bash
python house_price_regression.py
```

3. The script will train the model, print all evaluation metrics, save `model_evaluation.png`, and save the trained model as `house_price_model.pkl`.
4. It will then prompt you in the terminal:

```
Enter square footage: 2000
Enter number of bedrooms: 3
Enter number of bathrooms: 2

Predicted price: $455,434.70
```

---

## 📁 Project Structure

```
house_price_regression.py           # main script — training, evaluation, prediction
house_price_regression_dataset.csv  # dataset
house_price_model.pkl               # saved trained model
model_evaluation.png                # saved evaluation plots
```

---

## 🔮 Possible Future Improvements

- Incorporate more features (`Year_Built`, `Lot_Size`, `Garage_Size`, `Neighborhood_Quality`) to improve prediction accuracy further.
- Try more advanced regression models (e.g. Random Forest Regressor, Gradient Boosting) and compare performance against the baseline Linear Regression model.
- Add cross-validation instead of a single train/test split for a more robust accuracy estimate.
- Build a simple web interface for entering house details instead of using the command line.
