# importing required modules
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import shutil
import tensorflow
from PIL import Image
from tensorflow.keras.preprocessing import image

# model import 
model = tensorflow.keras.models.load_model("model_final.h5")
# classes label
index_to_label = {
    0: "Alstonia Scholaris",
    1: "Apple",
    2: "Arjun",
    3: "No Leaves Detected",
    4: "Bael",
    5: "Basil",
    6: "Blueberry",
    7: "Cherry",
    8: "Chinar",
    9: "Corn",
    10: "Gauva",
    11: "Grape",
    12: "Grapevine",
    13: "Jamun",
    14: "Jatropha",
    15: "Lemon",
    16: "Mango",
    17: "Peach",
    18: "Pepper",
    19: "Pomegranate",
    20: "Pongamia Pinata",
    21: "Potato",
    22: "Rasberry",
    23: "Soybean",
    24: "Strawberry",
    25: "Tomato"
}

# app initialization
app = FastAPI()#

# welcome message
@app.get("/")
def index():
    return{"Welcome to PAKPLANTS"}

# prediction endpoint
@app.post("/plant")
def predict(file: UploadFile =File(...)):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    Plant_image = Image.open(file.filename)
    # resize image
    Plant_image = Plant_image.resize((224, 224))
    # convert image to numpy array
    img_array = image.img_to_array(Plant_image)
    img_array=img_array/255
    # reshape image
    image_np = np.expand_dims(img_array, axis=0)
    # predict image
    prediction =model.predict(image_np)
    prediction=np.argmax(prediction, axis=1)
    # return prediction
    prediction=index_to_label[prediction[0]]
    return{prediction}

# run server
if __name__ == "__main__":
    uvicorn.run(app, debug=True)


#        curl -X 'POST' \
#       'http://localhost:8000/percent' \
#      -H 'accept: application/json' \
#     -H 'Content-Type: multipart/form-data' \
#    -F 'file=@wh0051.jpg;type=image/jpeg'