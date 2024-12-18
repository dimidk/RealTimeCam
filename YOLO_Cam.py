
import numpy as np
import time
import cv2
import torch
import math

from picamera2 import Picamera2

#from sympy.printing.pretty.pretty_symbology import annotated
from ultralytics import YOLO
from ultralytics import settings


# View all settings
print(settings)

# Return a specific setting
value = settings["runs_dir"]

import os
print("file exists?", os.path.exists("SampleVideo_LowQuality.mp4"))

url = "SampleVideo_LowQuality.mp4"
url_1 = "TrafficPolice.mp4"


#for real time applications where speed is more important like live video
#than high precision then n or s is more suitable
#Also hardware constraints are working better with n or s and with low energy
#consumptions
#So, pre-trained models like n or s are more suitable for raspberry deploy

model = YOLO('yolov8n.pt') 
classNames = model.names
print(classNames)

cam = Picamera2()
cam.configure(cam.create_preview_configuration(raw={"size":(1640,1232)}, main={"format":'RGB888', "size": (640, 480)}))
cam.start()
time.sleep(2)


while True:
    
    frame = cam.capture_array()
    
    if frame is None:
        break
    frame = cv2.flip(frame,-1)
    
    #results = model.predict(frame)
    results = model(frame,stream=True)
    #annotated_frame = results[0]
    
    for res in results:
        boxes = res.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])
            
            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (245, 0, 0)
            thickness = 2

            # put object details in box wich box has certain color and font
            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
    
    
    cv2.imshow('My YOLO v8 Object Detection',frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
#cap.release()
cap.stop()
cv2.destroyAllWindows()
