import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

COL_SEPAL_LENGTH = "sepal length (cm)"
COL_SEPAL_WIDTH = "sepal width (cm)"
COL_PETAL_LENGTH = "petal length (cm)"
COL_PETAL_WIDTH = "petal width (cm)"

X, y = load_iris(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
with open("accuracy.json", "w") as f:
    json.dump(accuracy, f)
