# from fastapi import FastAPI
# app = FastAPI()
# def simple_model(number:int):
#     if number % 2 ==0:
#         return "Cut reqemdir"
#     else:
#         return "Tek reqemdir"
    
# @app.get("/")
# def home():
#     return {"mesaj": "MLOps API-miz ishleyr"}
# @app.get ("/predict/{number}")
# def predict(number:int):
#     result=simple_model(number)
#     return{
#         "input":number,
#         "prediction":result
#     } 

import pandas as pd
from fastapi import FastAPI
app = FastAPI()

data={
    "model":["BMW","Mercedes","Toyota"],
    "qiymet":[50000,60000,123456788],
    "il":[2020,2021,2019]
}
df=pd.DataFrame(data)

@app.get("/")
def home():
    return {"mesaj":"data-merkezli MLOPS apimiz hazirdir!"}

@app.get("/get_data")
def get_all_data():
    return df.to_dict(orient="records")
@app.get("/average_price")
def get_avg_price():
    avg = df["qiymet"].mean()

    return {"ortalama_qiymet":avg} 