import numpy as np
import pandas as pd
import os
import torch
import cv2
import math
from PIL import Image

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
video_path = os.path.join(CURR_DIR, r"data\videos\burger_day_1.mp4")
model_path = os.path.join(CURR_DIR, r"runs\train\yolov5s_ufa\weights\best.pt")

model = torch.hub.load('Ultralytics/yolov5', 'custom', model_path, trust_repo=True).eval()
model.conf = 0.7
model.iou = 0.5

cap = cv2.VideoCapture(video_path)

color_dict = {0: (0,255,0),
             1: (255,175,80),
             2: (0,100,255),
             3: (255,215,0),
             4: (255,119,182)
             }

pol1 = (516/1280, 187/720)
pol2 = (250/1280, 498/720)
pol3 = (990/1280, 715/720)
pol4 = (1033/1280, 238/720)

def LineEquation(x, y, poly, start, end):
    x_a = poly[start][0]
    x_b = poly[end][0]
    y_a = poly[start][1]
    y_b = poly[end][1]
    return ( (x - x_a) / (x_b - x_a) ) - ( (y - y_a) / (y_b - y_a) )


tracking_obejcts = {}

center_points_prev_frame=[]


default_distance = 25
count = 0

while cap.isOpened():
    ret, frame = cap.read()


    y_res, x_res = frame.shape[0], frame.shape[1]
    polygone = np.array([(pol1[0] * x_res, pol1[1] * y_res), 
                         (pol2[0] * x_res, pol2[1] * y_res), 
                         (pol3[0] * x_res, pol3[1] * y_res), 
                         (pol4[0] * x_res, pol4[1] * y_res)], dtype=int)
    

    count += 1
    results = model(frame)
    df = results.pandas().xyxy[0]
    tups = list(df.itertuples(index=False))

    center_points_cur_frame = []

    for tup in tups:
        x = (int(tup[0]), int(tup[2]))
        y = (int(tup[1]), int(tup[3]))
        cx = int(x[0] - (x[0] - x[1]) / 2)
        cy = int(y[0] + (y[1] - y[0]) / 2)
        
        dist_x = ( cx - default_distance, cx + default_distance )
        dist_y = ( cy + default_distance, cy - default_distance )

        if tup[5] != 2 \
            and (LineEquation(cx, cy, polygone, 0, 1) < 0) \
            and (LineEquation(cx, cy, polygone, 1, 2) > 0) \
            and (LineEquation(cx, cy, polygone, 2, 3) < 0) \
            and (LineEquation(cx, cy, polygone, 3, 0) > 0):
            image = cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]), color_dict[tup[5]], 2)
            image = cv2.rectangle(image, (dist_x[0], dist_y[0]), (dist_x[1], dist_y[1]), (255,255,255), 1)
            image = cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)
            image = cv2.putText(image, tup[6], (x[0], y[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            center_points_cur_frame.append((cx, cy))

        image = cv2.polylines(image, [polygone.reshape((-1, 1, 2))], True, (255, 4, 0), 2)


    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_obejcts[track_id] = pt
                    track_id += 1
    else:
        tracking_obejcts_copy = tracking_obejcts.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_obejcts_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:        
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                
                # Update IDs position
                if distance < 20:
                    tracking_obejcts[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove id lost
            if not object_exists:
                tracking_obejcts.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_obejcts[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_obejcts.items():
        image = cv2.circle(image, pt, 5, (0,0,255),-1)
        image = cv2.putText(image, str(object_id), (pt[0],pt[1]-7),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

    cv2.imshow('frame', image)

    center_points_prev_frame = center_points_cur_frame.copy()

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()