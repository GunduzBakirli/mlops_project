import pandas as pd
from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()
iris=load_iris()
x=iris.data
y=iris.target
model = RandomForestClassifier()
model.fit(x, y)
@app.get('/')
def home():
    return {"mesaj":"Real ML modeli ile ishleyen API aktivdir!"}

@app.get("/predict")
def predict(sepal_l: float, sepal_w: float, petal_l: float, petal_w: float):
    prediction = model.predict([[sepal_l, sepal_w, petal_l, petal_w]])
    label = iris.target_names[prediction[0]]