
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import dataLoader

model = LinearRegression()
model.fit(dataLoader.X_train, dataLoader.y_train)
y_pred = model.predict(dataLoader.X_test)

mse = mean_squared_error(dataLoader.y_test, y_pred)
r2 = r2_score(dataLoader.y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
