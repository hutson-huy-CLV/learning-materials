import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
# from ydata_profiling import ProfileReport
#%%
df = pd.read_csv("D:\Hutson\learning-materials\AI&ML\Khóa 12-02AIMLDLCV nâng cao\Class\Datasets\car.csv")

df['brandname'] = df['car name'].str.split(' ').str.get(0)

unused_columns = ['car name']
df = df.drop(unused_columns, axis=1)
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
#%%

target = 'mpg'

x = df.drop(target, axis=1)
y = df[target]

# profile = ProfileReport(df, title="Car Report", explorative=True)
# profile.to_file("D:\Hutson\learning-materials\AI&ML\Khóa 12-02AIMLDLCV nâng cao\Class\car.html")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


num_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
nom_columns = ['brandname']


num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, num_columns),
    ("nom_features", nom_transformer, nom_columns),
])


reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
for i, j in zip(y_test, y_predict):
    print("Actual: {}. Predict: {}".format(i, j))

### r2 score is most common metric for regression (the higher the better)
print(reg.score(x_test, y_test))
