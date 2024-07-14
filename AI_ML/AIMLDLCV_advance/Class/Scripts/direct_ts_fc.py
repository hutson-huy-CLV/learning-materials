import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_direct_data(data, window_size, target_size):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    i = 0
    while i < target_size:
        data["target_{}".format(i)] = data["co2"].shift(-i-window_size)
        i += 1
    data = data.dropna(axis=0)
    return data


data = pd.read_csv("co2.csv")
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# plt.show()
window_size = 5
num_targets = 3
data = create_direct_data(data, window_size, num_targets)

targets = ["target_{}".format(i) for i in range(num_targets)]
x = data.drop(["time"] + targets, axis=1)
y = data[targets]
train_size = 0.8
num_samples = len(data)
x_train = x[:int(train_size*num_samples)]
y_train = y[:int(train_size*num_samples)]
x_test = x[int(train_size*num_samples):]
y_test = y[int(train_size*num_samples):]

models = [LinearRegression() for _ in range(num_targets)]
r2 = []
mae = []
mse = []
for i, model in enumerate(models):
    model.fit(x_train, y_train["target_{}".format(i)])
    y_predict = model.predict(x_test)
    r2.append(r2_score(y_test["target_{}".format(i)], y_predict))
    mse.append(mean_squared_error(y_test["target_{}".format(i)], y_predict))
    mae.append(mean_absolute_error(y_test["target_{}".format(i)], y_predict))

print("R2 score: {}".format(r2))
print("Mean absolute error: {}".format(mae))
print("Mean squared error: {}".format(mse))
