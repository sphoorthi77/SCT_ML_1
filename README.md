# House Price Prediction — Linear Regression

Predicts house prices from **square footage**, **number of bedrooms**, and **number of bathrooms** using scikit-learn's `LinearRegression`.

## Results on this dataset
- R² ≈ 0.987
- RMSE ≈ $29,093

## Files
- `house_price_regression.py` — main script (load data, train, evaluate, predict, save model/plot)
- `house_price_regression_dataset.csv` — dataset
- `requirements.txt` — Python dependencies
- `house_price_model.pkl` — saved trained model (generated after running)
- `actual_vs_predicted.png` — evaluation plot (generated after running)

---

## 1. Run it in VS Code

1. **Install prerequisites** (if not already installed):
   - [Python 3.9+](https://www.python.org/downloads/)
   - [VS Code](https://code.visualstudio.com/)
   - VS Code extension: **Python** (by Microsoft)

2. **Open the project folder in VS Code**
   - `File → Open Folder…` → select this project's folder.

3. **Create and activate a virtual environment** (recommended)
   Open the VS Code terminal (`` Ctrl+` `` / `` Cmd+` ``) and run:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
   VS Code may prompt "Select Interpreter" — choose the `venv` one.

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the script**
   - Click the ▶ "Run Python File" button (top-right), or in the terminal:
   ```bash
   python house_price_regression.py
   ```

6. **Check outputs**
   - Console prints coefficients, MAE/MSE/RMSE/R², and a sample prediction.
   - `house_price_model.pkl` (saved model) and `actual_vs_predicted.png` (chart) appear in the folder — click the PNG in VS Code's Explorer to preview it.

---

## 2. Push the project to Git / GitHub

1. **Create a `.gitignore`** (avoid committing the venv):
   ```bash
   echo "venv/" >> .gitignore
   echo "__pycache__/" >> .gitignore
   ```

2. **Initialize git and make the first commit** (in the VS Code terminal, inside the project folder):
   ```bash
   git init
   git add .
   git commit -m "Add house price linear regression model"
   ```

3. **Create a new empty repository on GitHub**
   - Go to https://github.com/new, give it a name (e.g. `house-price-regression`), leave it empty (no README/license), click **Create repository**.

4. **Link and push**
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
   (Sign in via GitHub in VS Code's Source Control panel if prompted, or use a personal access token if asked for a password.)

5. **Alternative: use VS Code's built-in Git UI**
   - Open the **Source Control** icon in the sidebar (`Ctrl+Shift+G`).
   - Click **Initialize Repository**.
   - Stage changes (`+`), write a commit message, click ✔ **Commit**.
   - Click **Publish Branch** → choose GitHub → pick public/private → done. VS Code creates the remote repo and pushes automatically.

That's it — your model code, dataset, and results are now versioned on GitHub.
