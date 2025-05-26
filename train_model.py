from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load Iris dataset
X, y = load_iris(return_X_y=True)

# Initialize and train the Logistic Regression model
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

print("Model trained")

