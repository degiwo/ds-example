from sklearn.datasets import load_iris
import seaborn as sns

COL_SEPAL_LENGTH = "sepal length (cm)"
COL_SEPAL_WIDTH = "sepal width (cm)"
COL_PETAL_LENGTH = "petal length (cm)"
COL_PETAL_WIDTH = "petal width (cm)"

X, y = load_iris(return_X_y=True, as_frame=True)

sns.scatterplot(
    data=X,
    x=COL_PETAL_LENGTH,
    y=COL_SEPAL_WIDTH,
    hue=y,
)
