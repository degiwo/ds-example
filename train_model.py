from sklearn.datasets import load_iris
import seaborn as sns

X, y = load_iris(return_X_y=True, as_frame=True)

sns.scatterplot(
    data=X,
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue=y,
)
