import logging
from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier # RandomForest-u DecisionTree ilə əvəz etdik
import time

# 1. Logging konfiqurasiyası (Olduğu kimi qalır, fayla və terminala yazır)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("MLOps-v2")

app = FastAPI()

# 2. Model v2.0 hazırlığı
iris = load_iris()
# DecisionTreeClassifier daha sürətli və sadə qərar ağacları qurur
model = DecisionTreeClassifier() 
model.fit(iris.data, iris.target)
logger.info("Model v2.0 (Decision Tree) uğurla öyrədildi və yükləndi.")

@app.get("/")
def home():
    logger.info("Home səhifəsinə giriş edildi.")
    return {"status": "aktiv", "model_version": "v2.0"}

@app.get("/predict")
def predict(sepal_l: float, sepal_w: float, petal_l: float, petal_w: float):
    start_time = time.time()
    
    # Proqnoz
    input_data = [[sepal_l, sepal_w, petal_l, petal_w]]
    prediction_index = model.predict(input_data)[0]
    label = str(iris.target_names[prediction_index])
    
    process_time = time.time() - start_time
    
    # LOGLAMA: Artıq model versiyasını da loglarda qeyd edirik
    logger.info(f"VERSION: v2.0 | SORĞU: {input_data} | CAVAB: {label} | VAXT: {process_time:.4f} san")
    
    return {
        "model_versiyası": "v2.0",
        "təxmin_edilən_növ": label,
        "process_time": f"{process_time:.4f}s"
    }