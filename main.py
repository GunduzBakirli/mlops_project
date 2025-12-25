import pandas as pd
from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

@app.get("/")
def home():
    return {"status": "aktiv"}

@app.get("/predict")
def predict(sepal_l: float, sepal_w: float, petal_l: float, petal_w: float):
    data = [[sepal_l, sepal_w, petal_l, petal_w]]
    prediction = model.predict(data)
    
    index = int(prediction[0])
    label = str(iris.target_names[index])
    
    return {
        "təxmin_edilən_növ": label
    }