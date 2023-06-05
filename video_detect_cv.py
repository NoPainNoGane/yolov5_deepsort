import numpy as np
import pandas as pd
import os
import torch
import cv2
import math
from vidgear.gears import WriteGear


CURR_DIR = os.path.dirname(os.path.realpath(__file__))
video_path = os.path.join(CURR_DIR, r"data\videos\2023-05-18 16-31-26.mp4")
model_path = os.path.join(CURR_DIR, r"runs\train\yolov5s_ufa2\weights\best.pt")
#MODEL PARAMETERS
model = torch.hub.load('Ultralytics/yolov5', 'custom', model_path, trust_repo=True).eval()
model.conf = 0.45
model.iou = 0.5



color_dict = {0: (0,255,0),
             1: (255,175,80),
             2: (0,100,255),
             3: (255,215,0),
             4: (255,119,182)
             }


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       print('x = %d, y = %d'%(x, y))


def LineEquation(x, y, poly, start, end):
    x_a = poly[start][0]
    x_b = poly[end][0]
    y_a = poly[start][1]
    y_b = poly[end][1]
    return ( (x - x_a) / (x_b - x_a) ) - ( (y - y_a) / (y_b - y_a) )

    
pol1 = (516/1280, 187/720)
pol2 = (250/1280, 498/720)
pol3 = (990/1280, 715/720)
pol4 = (1045/1280, 238/720)
ids = list(range(512))

frames_persec = 30
hours_perframe = 1 / 60 / 60 / frames_persec
dist_real = 0.0266#26.6m
default_distance = 35#euclid metric

count = 0
tracking_obejcts = {}
center_points_prev_frame = []


#Result Dataframe
getResult = True
df_results = pd.DataFrame(columns=['frame', 'id', 'x', 'y', 'lon', 'lat', 'class', 'speed'])

cap = cv2.VideoCapture(video_path)
#VIDEO REC
video_rec = False
output_params = {"-vcodec": "h264_vaapi", "-crf": 0, "-preset": "fast", "-input_framerate": 30}
if video_rec: writer = WriteGear(output=os.path.join(CURR_DIR, 'out.mp4'), logging=True, **output_params)


while cap.isOpened():
    ret, frame = cap.read()#(808, 1440, 3)

    y_res, x_res = frame.shape[0], frame.shape[1]
    polygone = np.array([(pol1[0] * x_res, pol1[1] * y_res), 
                         (pol2[0] * x_res, pol2[1] * y_res), 
                         (pol3[0] * x_res, pol3[1] * y_res), 
                         (pol4[0] * x_res, pol4[1] * y_res)], dtype=int)

    #middle points of left and right equations
    dist_cam_p = np.array([((pol1[0] + pol2[0]) / 2 * x_res, (pol1[1] + pol2[1]) / 2 * y_res), 
                         ((pol3[0] + pol4[0]) / 2 * x_res, (pol3[1] + pol4[1]) / 2 * y_res)],dtype=int)
    
    #euclid distance between middle of left and right points of equations
    dist_cam = math.hypot(dist_cam_p[0][0] - dist_cam_p[1][0], dist_cam_p[0][1] - dist_cam_p[1][1])

    km_perpixel = dist_real / dist_cam

    count += 1

    results = model(frame)
    df = results.pandas().xyxy[0]
    tups = list(df.itertuples(index=False))

    center_points_cur_frame = []
    image = frame.copy()
    
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
            center_points_cur_frame.append(((cx, cy), tup[6]))

        image = cv2.polylines(image, [polygone.reshape((-1, 1, 2))], True, (255, 4, 0), 2)

    image = cv2.putText(image, f"amount of cars: {len(center_points_cur_frame)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 255, 20), 2)

    if count <= 2:
        for pt, cls in center_points_cur_frame:
            for pt2, cls2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < default_distance:
                    speed_t = distance * km_perpixel / hours_perframe
                    tracking_obejcts[ids[0]] = pt, speed_t, cls
                    ids.remove(ids[0])

    else:
        tracking_obejcts_copy = tracking_obejcts.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()
        for object_id, (pt2, speed, cls2) in tracking_obejcts_copy.items():
            object_exists = False
            for pt, cls in center_points_cur_frame_copy:
                
                distance = math.hypot((pt2[0] - pt[0])*0.7, (pt2[1] - pt[1])*1.2)
                
                # Update IDs positionr
                if distance < default_distance:
                    speed_t = distance * km_perpixel / hours_perframe
                    tracking_obejcts[object_id] = pt, speed_t, cls
                    object_exists = True
                    if (pt, cls) in center_points_cur_frame:
                        center_points_cur_frame.remove((pt, cls))
                    continue

            # Remove id lost
            if not object_exists:
                tracking_obejcts.pop(object_id)
                ids.append(object_id)

        # Add new IDs found
        for pt, cls in center_points_cur_frame:
            tracking_obejcts[ids[0]] = pt, 0, cls
            ids.remove(ids[0])

    for object_id, (pt, speed, cls) in tracking_obejcts.items():
        image = cv2.circle(image, pt, 5, (0,0,255), -1)
        image = cv2.putText(image, f"{str(object_id)} s: {'%.2f'%speed} km/h", (pt[0] - 12, pt[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if getResult: df_results.loc[len(df_results.index)] = [count, object_id, pt[0], pt[1], 0, 0, cls, speed]

    image = cv2.resize(image, (1280,720), interpolation = cv2.INTER_AREA)
    if video_rec: writer.write(image)
    cv2.imshow('frame', image)
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', onMouse)

    center_points_prev_frame = center_points_cur_frame.copy()

    if cv2.waitKey(1) == ord('e'):
        break

if getResult: df_results.to_csv(os.path.join(CURR_DIR, 'out.csv'), index=False)

cap.release()
if video_rec: writer.close()
cv2.destroyAllWindows()