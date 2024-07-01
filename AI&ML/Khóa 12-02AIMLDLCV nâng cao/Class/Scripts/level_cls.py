import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import re

def filter_location(loc):
    result = re.findall("\,\s[A-Z]{2}$", loc)
    if len(result):
        return result[0][2:]
    else:
        return loc


data = pd.read_excel("final_project.ods", engine="odf", dtype=str)
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(), "title"),
    ("location", OneHotEncoder(), ["location"]),
    ("description", TfidfVectorizer(stop_words="english"), "description"),
    ("function", OneHotEncoder(), ["function"]),
    ("industry", TfidfVectorizer(), "industry"),
])

processed_data = preprocessor.fit_transform(x_train)
print(processed_data.shape)


