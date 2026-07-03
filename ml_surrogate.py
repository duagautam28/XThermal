import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt


df=pd.read_csv("thermal_dataset.csv")

print(df.head())
print(df.shape)
print("columns=",df.columns)
print(len(df))

X = df[["TDP","air_velocity","tim_conductivity"]]

Y = df[["thermal_resistance","junction_temperature"]]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

print('testing samples',len(X_test))
print('training samples',len(X_train))

linear_model=LinearRegression()
linear_model.fit(X_train,Y_train)

Y_prediction=linear_model.predict(X_test)

mae=mean_absolute_error(Y_test,Y_prediction)

mse=root_mean_squared_error(Y_test,Y_prediction)

r2= r2_score(Y_test,Y_prediction)

print("linear regression result")
print("MAE=", mae)
print("RMSE= ", mse)
print("R2 score=", r2)

rf_model=RandomForestRegressor(n_estimators=100,random_state=42)

rf_model.fit(X_train,Y_train)
Y_prediction_rf=rf_model.predict(X_test)

rf_mae=mean_absolute_error(Y_test,Y_prediction_rf)

rf_mse=root_mean_squared_error(Y_test,Y_prediction_rf)

rf_r2= r2_score(Y_test,Y_prediction_rf)

print("random forest result")
print("MAE=", rf_mae)
print("RMSE= ", rf_mse)
print("R2 score=", rf_r2)

comparison = pd.DataFrame({
    "Model": ["linear regression", "random forest"],
    "MAE": [mae, rf_mae],
    "RMSE": [mse, rf_mse],
    "R2 Score": [r2, rf_r2]
})

print("\nModel Comparison")
print(comparison)

imp=rf_model.feature_importances_

feature_imp=pd.DataFrame({
    "feature":X.columns,
    "importance":imp
})

print("\nFeature Importance")
print(feature_imp)

plt.figure(figsize=(6,4))
plt.bar(feature_imp["feature"],feature_imp["importance"])

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")

plt.tight_layout()

plt.show()