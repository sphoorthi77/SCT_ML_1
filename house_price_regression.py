import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib

DATA_PATH = "house_price_regression_dataset.csv"
df = pd.read_csv(DATA_PATH)

FEATURES = ["Square_Footage", "Num_Bedrooms", "Num_Bathrooms"]
TARGET = "House_Price"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Coefficients:")
for feature, coef in zip(FEATURES, model.coef_):
    print(f"  {feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}\n")

print("Regression Evaluation Metrics:")
print(f"  MAE  (Mean Absolute Error)      : {mae:.2f}")
print(f"  MSE  (Mean Squared Error)       : {mse:.2f}")
print(f"  RMSE (Root Mean Squared Error)  : {rmse:.2f}")
print(f"  R^2  (Coefficient of Determination): {r2:.4f}")
print(f"  -> On average, predictions are off by about ${mae:,.0f}.")
print(f"  -> The model explains {r2*100:.1f}% of the variation in house prices.\n")

bins = [0, 300000, 600000, 900000, np.inf]
labels = ["Low", "Medium", "High", "Very High"]

y_test_cat = pd.cut(y_test, bins=bins, labels=labels)
y_pred_cat = pd.cut(y_pred, bins=bins, labels=labels)

acc = accuracy_score(y_test_cat, y_pred_cat)
print(f"Price Category Accuracy (Low/Medium/High/Very High): {acc*100:.2f}%")
print("  -> This treats price ranges as classes, purely to give an accuracy/confusion matrix view.\n")

cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)
print("Confusion Matrix (rows = actual category, columns = predicted category):")
print(pd.DataFrame(cm, index=labels, columns=labels), "\n")

joblib.dump(model, "house_price_model.pkl")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.6, edgecolor="k", label="Predictions")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax1.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction Line")
ax1.set_xlabel("Actual Price ($)")
ax1.set_ylabel("Predicted Price ($)")
ax1.set_title("Actual vs Predicted House Prices")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.text(
    0.05, 0.95,
    f"R^2 = {r2:.3f}\nCloser dots to the red line\n= better predictions",
    transform=ax1.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

ax2 = axes[1]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax2, cmap="Blues", colorbar=False)
ax2.set_title(f"Price Category Confusion Matrix\n(Accuracy = {acc*100:.1f}%)")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
print("Plot saved as model_evaluation.png")

sqft = float(input("\nEnter square footage: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

user_input = pd.DataFrame({
    "Square_Footage": [sqft],
    "Num_Bedrooms": [bedrooms],
    "Num_Bathrooms": [bathrooms]
})

predicted_price = model.predict(user_input)[0]
print(f"\nPredicted price: ${predicted_price:,.2f}")
