import io
import os
import cv2
import uvicorn
import numpy as np
import nest_asyncio
import subprocess
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cvlib as cv
from cvlib.object_detection import draw_bbox
from fastapi import Form


app = FastAPI(title='Deploying an ML Model with FastAPI')

class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://serve/docs"


@app.post("/predict") 
def prediction(model: Model = Form(...), file: UploadFile = File(...)):

    # 1. VALIDATE INPUT FILE
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    
    # 2. TRANSFORM RAW IMAGE INTO CV2 image
    
    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())
    
    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    
    # 3. RUN OBJECT DETECTION MODEL
    
    # Run object detection
    bbox, label, conf = cv.detect_common_objects(image, model=model)

    # Clip bbox coordinates to image dimensions so cv2.rectangle/putText don't raise
    # ValueError: Value out of range (OpenCV 4.x strict bounds enforcement)
    h, w = image.shape[:2]
    bbox = [
        [max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)]
        for x1, y1, x2, y2 in bbox
    ]
    
    # Create image that includes bounding boxes and labels
    output_image = draw_bbox(image, bbox, label, conf)

    dir_name = "images_uploaded"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    # Save it in a folder within the server
    cv2.imwrite(f'images_uploaded/{filename}', output_image)
    
    
    # 4. STREAM THE RESPONSE BACK TO THE CLIENT
    
    # Open the saved image for reading in binary mode
    file_image = open(f'images_uploaded/{filename}', mode="rb")
    
    # Return the image as a stream specifying media type
    return StreamingResponse(file_image, media_type="image/jpeg")


nest_asyncio.apply()

# Set up for external access
host = "localhost"

# Spin up the server!    
uvicorn.run(app, host=host, port=7000)
