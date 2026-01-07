# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# 1. 加载数据集
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="price")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 周志华集成学习：XGBoost（梯度提升树）
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_mse = mean_squared_error(y_test, y_pred_xgb)

# 3. 周志华集成学习：LightGBM（直方图优化提升树）
lgb = LGBMRegressor(n_estimators=100, random_state=42)
lgb.fit(X_train, y_train)
y_pred_lgb = lgb.predict(X_test)
lgb_mse = mean_squared_error(y_test, y_pred_lgb)

# 4. 结果评估
print("✅ 周志华集成学习算法实践结果：")
print(f"XGBoost均方误差(MSE)：{xgb_mse:.2f}")
print(f"LightGBM均方误差(MSE)：{lgb_mse:.2f}")

# 5. 特征重要性（周志华集成学习特征分析）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
pd.Series(xgb.feature_importances_, index=X.columns).sort_values().plot(kind="barh", title="XGBoost特征重要性")
plt.subplot(1,2,2)
pd.Series(lgb.feature_importances_, index=X.columns).sort_values().plot(kind="barh", title="LightGBM特征重要性")
plt.tight_layout()
plt.savefig("周志华集成学习特征重要性.png")
plt.show()