# Assignment
# 데이터 로딩 -> 데이터 전처리 -> 타겟 및 독립변수 설정 -> 트레이닝/테스트 셋 설정
# -> 모델 선택 및 학습 -> 예측 수행 -> 성능 평가(mse,결정계수 R2 등) -> 시각화

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

#데이터 로딩
mpg = sns.load_dataset('mpg')
# print(mpg.info())
# print(mpg.tail(10))]

#데이터 전처리
mpg.dropna(subset=["horsepower"], inplace=True)
mpg.drop("name", axis=1, inplace=True)

encoder = OneHotEncoder(drop="first", sparse_output=False)
origin_encoded = encoder.fit_transform(mpg[["origin"]])

mpg[encoder.get_feature_names_out(["origin"])] = origin_encoded
mpg.drop(columns=["origin"], inplace=True)

target_variable = "mpg"
feature_variables = ["cylinders", "displacement", "horsepower", "weight",
                     "acceleration", "model_year", *encoder.get_feature_names_out(["origin"])]
X = mpg[feature_variables]
y = mpg[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)  # 🚀 NaN이 사라졌기 때문에 이제 정상적으로 실행됨

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE(평균제곱오차): {mse:.2f}")
print(f"R²(결정계수): {r2:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.8)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', linestyle='solid')
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.show()