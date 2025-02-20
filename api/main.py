from fastapi import FastAPI, File, UploadFile
import uvicorn   # just a server that runs the fastapi app
import numpy as np
from io import BytesIO    # class in the io module that allows you to read and write bytes-like objects in memory
from PIL import Image   # used to read images in python
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

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

    predictions  = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return{
        'class' : predicted_class,
        'confidence' : float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)