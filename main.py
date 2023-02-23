from fastapi import FastAPI, Request
import tensorflow as tf
import numpy as np

app = FastAPI()

CLASSES_NAMES = ["Bed", "Chair", "Sofa"]
model = tf.keras.models.load_model("model.h5")
img_height = 224
img_width = 224


def predict(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return (
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            CLASSES_NAMES[np.argmax(score)], 100 * np.max(score)
        )
    )


@app.post("/predict")
async def run_prediction(request: Request):
    data = await request.json()
    image_path = data.get("image_path")
    print(image_path)
    prediction = predict(image_path)
    return {"Prediction": prediction}


@app.get("/")
async def get():
    return "hello world"
