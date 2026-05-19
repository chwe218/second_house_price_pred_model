# Shanghai Second-Hand House Price Prediction 🏠

A lightweight regression project using real Shanghai housing data  
to predict second-hand house prices based on structural features.
This model can predict second-hand prices according to the input.

---

## Dataset
- Source: [Shanghai Second-Hand House Dataset (Kaggle)](https://www.kaggle.com/datasets/vnvile/shanghai-second-hand-house-dataset)
- Format: Parquet
- Unit: CNY

Main features:
- Area
- Room / Hall / Toilet count
- Year built
- Renovation
- Floor location
- District

Target:
- `listing_total_price` (total house price)

---

## Problem Statement
Predict the total price of a second-hand apartment  
given its structural characteristics.

---

## Methodology
- Data cleaning (minimal missing values removed)
- Feature selection (structural features only)
- OneHotEncoder for categorical and numeric features
- Train / Test split (80% / 20%)
- Model: Random Forest Regressor(Pipeline)

---

## Results 

| Metric | Value |
|------|------|
| MAE  | 128 |
| RMSE | 281 |
| R²   | 0.758 |

Interpretation:
- On average, predictions are within **≈ 128,000 CNY** of the true price.
- The model explains **75.8%** of the variance in housing prices.

---

## Key Insights
Feature importance analysis shows that:
- **Area**
- **Location (District)**
- **Year Built**

are the most influential factors in determining house prices.

---

## Installation & Usage

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/chwe218/second_house_price_pred_model.git
   cd second_house_price_pred_model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Make Predictions
Run the interactive prediction tool:
```bash
python predict.py
```

You'll be prompted to enter:
- Area (㎡)
- Number of rooms, halls, toilets
- Year built
- Renovation type (精装 / 简装 / 毛坯)
- District (e.g., 浦东, 闵行, 徐汇, etc.)
- Floor location (低楼层 / 中楼层 / 高楼层)

The model will return an estimated price in **万元** (10,000 CNY).

### Training (For Learning)
To retrain the model on the dataset:
```bash
python price_pred.py
```

---

## Improvements
- The model can still be optimized (hyperparameter tuning, feature engineering, try other algorithms)
- Cross-validation for more robust evaluation
- Add outlier detection and handling
- Test with other models (XGBoost, Gradient Boosting)
