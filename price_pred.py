import pandas as pd
df=pd.read_parquet(r"dataset.parquet")
df = df.loc[:, df.columns != 0]
print(df.columns.tolist())
# print(df.columns.tolist())
#选特征
features = [
    "area",
    "room_count",
    "hall_count",
    "toilet_count",
    "year_built",
    "renovation",
    "floor_location",
    "district"
]

X = df[features].copy()
y = df["listing_total_price"].copy()

# 缺失值

X = X.dropna(subset=["year_built", "floor_location"])
y = y.loc[X.index]

#  One-Hot 编码
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = [
    "renovation",
    "floor_location",
    "district"
]

numeric_features = [
    "area",
    "room_count",
    "hall_count",
    "toilet_count",
    "year_built"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",  # ✅ 预测时遇到新类别不会崩
                drop="first"
            ),
            categorical_features
        ),
        (
            "num",
            "passthrough",#不做任何变化直接通过
            numeric_features
        )
    ]
)
#训练模型

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)



from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("rf", RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


#评估
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_housing_model(y_true, y_pred, scale=1.0, unit="CNY"):
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("📊 Model Evaluation")
    print(f"MAE  : {mae / scale:.2f} ({unit})")
    print(f"RMSE : {rmse / scale:.2f} ({unit})")
    print(f"R²   : {r2:.3f}")


evaluate_housing_model(y_test, y_pred, scale=1, unit="CNY")



import joblib

joblib.dump(model, "house_price_rf_pipeline.pkl")
print("✅ Pipeline 模型已保存")

