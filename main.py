from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# 1. Modeli funksiyadan kenarda bir defe hazırlayırıq
iris = load_iris()
X = iris.data
y = iris.target
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

@app.get("/")
def home():
    return {"status": "aktiv", "model": "RandomForest"}

@app.get("/predict")
def predict(sepal_l: float, sepal_w: float, petal_l: float, petal_w: float):
    # 2. Giriş məlumatını modelin formatına salırıq
    input_data = [[sepal_l, sepal_w, petal_l, petal_w]]
    
    # 3. Proqnoz alırıq
    prediction_index = model.predict(input_data)[0]
    
    # 4. İndeksi çiçək adına çeviririk
    result = str(iris.target_names[prediction_index])
    
    # 5. Nəticəni qaytarırıq
    return {
        "təxmin_edilən_növ": result,
        "istifadə_olunan_ölçülər": [sepal_l, sepal_w, petal_l, petal_w]
    }