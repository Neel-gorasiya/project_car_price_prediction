# project_car_price_prediction



## 🔍 Overview

The resale price of a car depends on dozens of factors — age, fuel type, kilometres driven, number of previous owners, and more. Estimating the right price manually is difficult and often inaccurate.

This project builds a complete **end-to-end Machine Learning pipeline** that:

- Explores and visualises the dataset to uncover hidden patterns
- Engineers meaningful features such as Car Age from the manufacture year
- Trains and compares four regression models
- Selects the best model (Random Forest) achieving **~96% R² accuracy**
- Allows prediction of any car's resale price by simply passing its attributes

This is a full data science project — from raw CSV to final predictions — ideal as a portfolio project, academic submission, or learning reference for ML beginners.



## 📦 Dataset

| Property | Details |
|----------|---------|
| Source | Kaggle — Car Price Dataset |
| Records | 301 cars |
| Features | 9 (including target) |
| Target | `Selling_Price` in Indian Rupees (Lakhs) |
| Missing Values | None 


---

## ⚙️ How It Works

The notebook is organised into 11 clearly labelled sections:

**1. Import Libraries**
All required libraries are imported — NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

**2. Load & Explore the Dataset**
The dataset is loaded and inspected for shape, data types, and missing values.

**3. Exploratory Data Analysis (EDA)**
Seven visualisations explore selling price distribution, fuel type and transmission breakdowns, scatter plots of numerical features, box plots, and year-wise price trends.

**4. Feature Engineering & Preprocessing**
- `Car_Age` is derived from the Year column (2024 − Year)
- `Car_Name` and `Year` are dropped as they are replaced or irrelevant
- Categorical columns are encoded using Label Encoding
- Features are scaled using StandardScaler for linear models

**5. Correlation Analysis**
A heatmap identifies `Present_Price` and `Car_Age` as the strongest predictors of selling price.

**6. Model Building**
Four models are trained and evaluated:
- Linear Regression (baseline)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

**7. Model Comparison**
All models are compared on MAE, RMSE, and R² score with 5-fold cross-validation.

**8. Best Model Deep Dive**
Random Forest is analysed with actual vs. predicted plots, residual plots, and feature importance charts.

**9. Hyperparameter Tuning**
GridSearchCV tunes the Random Forest over `n_estimators`, `max_depth`, and `min_samples_split`.

**10. Predict New Car Price**
A sample prediction demonstrates how to pass any car's attributes and get an estimated selling price.

**11. Summary & Conclusions**
Key findings, model leaderboard, and actionable business insights are presented.

---

## 📊 Results

| Model | R² Score | MAE (₹ Lakhs) | RMSE (₹ Lakhs) |
|-------|----------|----------------|-----------------|
| 🏆 **Random Forest** | **~0.96** | **~0.80** | **~1.10** |
| Gradient Boosting | ~0.95 | ~0.90 | ~1.20 |
| Decision Tree | ~0.91 | ~1.10 | ~1.60 |
| Linear Regression | ~0.86 | ~1.50 | ~2.00 |

> Random Forest Regressor is the best-performing model with approximately **96% accuracy** on unseen test data.

### Top Predictive Features

| Rank | Feature | Impact |
|------|---------|--------|
| 1 | `Present_Price` | Highest |
| 2 | `Car_Age` | High |
| 3 | `Driven_kms` | Medium |
| 4 | `Transmission` | Medium |
| 5 | `Fuel_Type` | Low–Medium |

---

## 👥 Who Is This For

| Audience | How It Helps |
|----------|-------------|
| 🎓 Students & Beginners | A complete, well-commented ML project to learn from start to finish |
| 📊 Data Science Learners | Covers EDA, feature engineering, model selection, and tuning in one notebook |
| 💼 Job Seekers | A clean portfolio project showcasing regression and ensemble method skills |
| 🚘 Car Buyers & Sellers | Understand what drives resale value and get a data-backed price estimate |
| 🏢 Automobile Businesses | Can be extended into a pricing tool or customer-facing web application |
| 🔬 Researchers | A clean baseline model and dataset for further experimentation |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| Python 3.8+ | Core programming language |
| NumPy | Numerical computations |
| Pandas | Data loading, cleaning, and transformation |
| Matplotlib | Base plotting and charts |
| Seaborn | Statistical visualisations |
| Scikit-learn | ML models, preprocessing, evaluation, and hyperparameter tuning |
| Jupyter Notebook | Interactive development environment |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
```

### 2. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the Notebook
```bash
jupyter notebook
```

Open `Car_Price_Prediction.ipynb` and run all cells from top to bottom.

### 4. Predict a Custom Car Price
Go to **Section 10** in the notebook and modify the `sample_car` dictionary with your own car's details to get an instant price prediction.

---

## 💡 Key Insights

- **Present Price** is the strongest predictor — newer, more expensive cars retain resale value better
- **Car Age** has a strong negative effect — older cars consistently sell for less
- **Automatic transmission** cars command significantly higher resale prices than manual ones
- **Diesel cars** have a higher average resale value compared to petrol or CNG
- **More previous owners** always reduces the selling price
- **Higher driven kilometres** lowers resale value, though the relationship is non-linear

---

## 🔮 Future Improvements

- [ ] Add more features: brand reputation, city of sale, accident history
- [ ] Experiment with XGBoost, LightGBM, and CatBoost for better accuracy
- [ ] Deploy as a web application using Flask or Streamlit
- [ ] Build a REST API for business integration
- [ ] Train on a larger dataset for improved generalisation
- [ ] Add SHAP values for model explainability

