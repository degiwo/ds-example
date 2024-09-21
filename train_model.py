from sklearn.datasets import load_iris
from matplotlib.axes._axes import Axes
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

COL_SEPAL_LENGTH = "sepal length (cm)"
COL_SEPAL_WIDTH = "sepal width (cm)"
COL_PETAL_LENGTH = "petal length (cm)"
COL_PETAL_WIDTH = "petal width (cm)"

X, y = load_iris(return_X_y=True, as_frame=True)

def plot(x) -> Axes:
    return sns.scatterplot(
        data=X,
        x=x,
        y=COL_SEPAL_WIDTH,
        hue=y,
    )

ax = plot(x=COL_SEPAL_LENGTH)
plt.savefig("iris_scatterplot.png")

x = st.selectbox(
    "x column", (COL_SEPAL_LENGTH, COL_SEPAL_WIDTH)
)

ax = plot(x)
st.pyplot(ax.figure, clear_figure=True)
