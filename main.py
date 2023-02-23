from fastapi import FastAPI, Request
import tensorflow as tf
import numpy as np

app = FastAPI()


def predict(image_path):
    return 0


@app.post("/predict")
async def run_prediction(request: Request):
    data = await request.json()
    image = data.get("image")
    prediction = predict(image)
    return {"Prediction": prediction}


@app.get("/")
async def get():
    return "hello world"
