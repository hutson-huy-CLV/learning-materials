import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTEN
import re

def filter_location(loc):
    result = re.findall("\,\s[A-Z]{2}$", loc)
    if len(result):
        return result[0][2:]
    else:
        return loc


data = pd.read_excel(r'Datasets\final_project.ods', engine="odf", dtype=str)
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]

ros = SMOTEN(random_state=42, k_neighbors=2, sampling_strategy={
    "managing_director_small_medium_company": 500,
    "specialist": 500,
    "director_business_unit_leader": 500,
    "bereichsleiter": 1000
})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(y_train.value_counts())
print("------------")
x_train, y_train = ros.fit_resample(x_train, y_train)
print(y_train.value_counts())
exit(0)

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(), "industry"),
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", SelectPercentile(chi2, percentile=5)),
    ("model", RandomForestClassifier())
])


cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))
#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.63      0.18      0.28       185
#          director_business_unit_leader       1.00      0.19      0.32        16
#                    manager_team_leader       0.66      0.76      0.71       518
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.86      0.93      0.89       889
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.78      1615
#                              macro avg       0.53      0.34      0.37      1615
#                           weighted avg       0.77      0.78      0.75      1615


