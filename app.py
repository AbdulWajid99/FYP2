from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
import shutil

app = FastAPI()

lower_green = np.array([36 ,0 , 0])
upper_green = np.array([102,255,255])

@app.get("/index")
def index():
    return "Welcome to PAKPLANTS"

@app.post("/percent")
def mask(file: UploadFile =File(...)):
    with open(f'{file.filename}' , "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    img= cv2.imread(file.filename) 
    grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)
    mask= cv2.inRange(grid_HSV, lower_green, upper_green)
    green_perc = (np.sum(mask) / np.size(mask))/255
    green_perc = green_perc*100
    return round(green_perc,3)


