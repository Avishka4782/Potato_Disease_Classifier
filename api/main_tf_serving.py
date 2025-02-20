from fastapi import FastAPI, File, UploadFile
import uvicorn   # just a server that runs the fastapi app
import numpy as np
from io import BytesIO    # class in the io module that allows you to read and write bytes-like objects in memory
from PIL import Image   # used to read images in python
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8502/v1/models/potatoes_model:predict"


MODEL = tf.keras.models.load_model("../trained_model/2.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, Server is Alive"
def read_file_as_image(data) -> np.ndarray:
     image = np.array(Image.open(BytesIO(data)))
     return image

@app.post("/predict")
async def upload(
    file: UploadFile = File(...)
):
    image =read_file_as_image(await file.read()) 
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class =  CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return{
        "class" : predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)