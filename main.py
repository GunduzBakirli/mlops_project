import logging
from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import time

# 1. Logging konfiqurasiyası
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()] # Nəticələri birbaşa terminalda göstərsin
)
logger = logging.getLogger("MLOps-API")

app = FastAPI()

# 2. Model hazırlığı
iris = load_iris()
model = RandomForestClassifier(n_estimators=10)
model.fit(iris.data, iris.target)
logger.info("ML Modeli uğurla öyrədildi və yükləndi.")

@app.get("/")
def home():
    logger.info("Home səhifəsinə giriş edildi.")
    return {"status": "aktiv"}

@app.get("/predict")
def predict(sepal_l: float, sepal_w: float, petal_l: float, petal_w: float):
    start_time = time.time() # Hesablama vaxtını ölçmək üçün
    
    # Proqnoz
    input_data = [[sepal_l, sepal_w, petal_l, petal_w]]
    prediction_index = model.predict(input_data)[0]
    label = str(iris.target_names[prediction_index])
    
    process_time = time.time() - start_time
    
    # LOGLAMA: Kim nəyi soruşdu və cavab nə oldu?
    logger.info(f"SORĞU: {input_data} | CAVAB: {label} | VAXT: {process_time:.4f} san")
    
    return {
        "təxmin_edilən_növ": label,
        "process_time": f"{process_time:.4f}s"
    }