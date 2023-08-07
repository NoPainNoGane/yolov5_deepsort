import numpy as np
import pandas as pd
import os
import torch
import cv2
import math
import m3u8
import urllib.request
import matplotlib.path as mplPath

from vidgear.gears import WriteGear

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

online = False

model_path = os.path.join(CURR_DIR, r"runs\train\yolov5s_ufa2\weights\best.pt")
video_path = os.path.join(CURR_DIR, r"data\videos\gleb_move.mp4")

playlist = "http://136.169.226.59/1-4/tracks-v1/mono.m3u8?token=32abe32a935349d0843a26d0988bdce1"
videoLink = os.path.dirname(playlist) + '/'

#MODEL PARAMETERS
model = torch.hub.load('Ultralytics/yolov5', 'custom', model_path, trust_repo=True).eval()
model.conf = 0.45
model.iou = 0.5

#Result Dataframe
getResult = True
df_results = pd.DataFrame(columns=['frame', 'id', 'x', 'y', 'lat', 'long', 'class', 'speed', 'lane'])
color_dict = {0: (0,255,0),
             1: (255,175,80),
             2: (0,100,255),
             3: (255,215,0),
             4: (255,119,182)
             }

colors_traffic = [
    (10, 10, 242),
    (10, 211, 242),
    (0, 247, 87),
    (243, 247, 0),
    (247, 103, 0),
    (247, 0, 169),
    (255, 255, 255)
]

#VIDEO REC
video_rec = False
output_params = {"-vcodec": "h264_vaapi", "-crf": 0, "-preset": "fast", "-input_framerate": 25}
if video_rec: writer = WriteGear(output=os.path.join(CURR_DIR, 'out.mp4'), logging=True, **output_params)


def download_files(local_files):
    """
    super cool function that could
    sequently download ts files from m3u8 playlist
    """
    m3u8_obj = m3u8.load(playlist)
    ts_segments_str = str(m3u8_obj.segments)
        
    for line in ts_segments_str.splitlines():
        if ".ts" in line:
            server_file_path = os.path.join(videoLink, line)
            file_name = line[line.rfind('/') + 1:line.find('?')]
            local_file_path = os.path.join(CURR_DIR, "video_files", file_name)
            if not local_file_path in local_files:
                local_files.append(local_file_path)
                urllib.request.urlretrieve(server_file_path, local_file_path)
    return local_files


def onMouse(event, x, y, flags, param):
    #EVENT
    if event == cv2.EVENT_LBUTTONDOWN:
       print('x = %d, y = %d'%(x, y))


def getRealcoords(pt, isItGPS=False):
    m_ext = np.array([[ 1.34649732e+00, 2.63158080e+00, 5.17662265e+02],
                    [-2.07245855e-01, -9.66656408e-01, 7.39496589e+02],
                    [-1.18351294e-03, 1.67921001e-03, 1.00000000e+00]])
    
    gps_ext = np.array([[ 2.10302973e+01, -1.22791129e+01, -4.63979104e+02],
                        [-4.87218236e+00,  5.07541375e+00, -1.72847276e+01],
                        [-6.45725593e-03, -1.15589441e-02, 1.00000000e+00]])
    
    vector = np.array([pt[0], pt[1], 1], dtype=np.float64)
    if isItGPS:
        result = np.linalg.solve(gps_ext, vector)
    else:
        result = np.linalg.solve(m_ext, vector)
    
    result = result / result[-1]
    return result[:2].astype(np.float64)


x_res, y_res = (1920, 1080)# RESOLUTION
x_resized, y_resized = (1280, 720) # NEW RESIZED RESOLUTION

#devided values got from new resolution (thant's why I'm making a devision)
pol5 = (650/x_resized, 203/y_resized)
pol1 = (554/x_resized, 237/y_resized)
pol2 = (343/x_resized, 490/y_resized)
pol3 = (934/x_resized, 693/y_resized)
pol4 = (978/x_resized, 278/y_resized)
pol6 = (887/x_resized, 220/y_resized)


polygon = np.array([(pol5[0] * x_res, pol5[1] * y_res),
                    (pol1[0] * x_res, pol1[1] * y_res), 
                    (pol2[0] * x_res, pol2[1] * y_res),
                    (pol3[0] * x_res, pol3[1] * y_res),
                    (pol4[0] * x_res, pol4[1] * y_res),
                    (pol6[0] * x_res, pol6[1] * y_res),], dtype=int)#points for polygone in the center
poly_path = mplPath.Path(polygon)

traffic_lanes = np.array([[(610, 674), (552, 746), (1348, 1004), (1369, 870)],
                        [(673, 595), (610, 674), (1369, 870), (1390, 735)],
                        [(726, 528), (673, 595), (1390, 735), (1404, 645)],
                        [(775, 467), (726, 528), (1404, 645), (1417, 562)],
                        [(817, 414), (775, 467), (1417, 562), (1429, 485)],
                        [(859, 362), (817, 414), (1429, 485), (1439, 421)],
                        [(961, 304), (930, 360), (1335, 400), (1339, 340)]],dtype=int)
traffic_lanes_path = [mplPath.Path(lane) for lane in traffic_lanes]

ids = list(range(512))
frames_persec = 25
hours_perframe = 1 / 60 / 60 / frames_persec
default_distance = 25#euclid metric


def main():
    count = 0
    tracking_objects = {}
    center_points_prev_frame = []

    if online:
        local_files = download_files([])
        del_file = None

    optimize = \
        [-0.44122265390547605,0.21375771203025706,-0.008192271315560882,-0.00143372092373238,-0.020766579977952507,1028.4401269037626,706.6902157673671,1551.0929328383193,1138.611383662474]
    camera_matrix = np.array([[optimize[7], 0.00000000e+00, optimize[5]],
                            [0.00000000e+00, optimize[8], optimize[6]],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_coefs = np.array([optimize[0], optimize[1], optimize[2], optimize[3], optimize[4]])

    while True:
        if online:
            local_file = local_files[0]
            cap = cv2.VideoCapture(local_file)
            if del_file:
                os.remove(del_file)
        else:
            cap = cv2.VideoCapture(video_path)
        
        file_frames = 0
        while cap.isOpened():
            
            ret, frame = cap.read()
            file_frames += 1
            if file_frames == 1: continue

            if ret == True:
                #reduction of distortion
                frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h,  w = frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
                frame = cv2.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

                    if tup[5] != 2 and poly_path.contains_point((cx, cy)):

                        gps_coord = getRealcoords((cx, cy), True)

                        image = cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]), color_dict[tup[5]], 2)# object rectangle
                        image = cv2.rectangle(image, (dist_x[0], dist_y[0]), (dist_x[1], dist_y[1]), (255,255,255), 1)# euclid metric rectangle
                        image = cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)# center of object
                        image = cv2.putText(image, tup[6], (x[0], y[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # label of object
                        image = cv2.putText(image, str(gps_coord[0]) +" "+ str(gps_coord[1]), (x[0] - 30, y[0] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)# GPS label of object
                        center_points_cur_frame.append(((cx, cy), tup[6]))

                image = cv2.polylines(image, [polygon.reshape((-1, 1, 2))], True, (255, 4, 0), 2)
                
                for k, lane in enumerate(traffic_lanes):
                    image = cv2.polylines(image, [lane.reshape((-1, 1, 2))], True, colors_traffic[k], 2)
                    image = cv2.putText(image, str(k), (lane[1][0] + 15, lane[1][1]), cv2.LINE_4, 1, colors_traffic[k], 2)

                image = cv2.putText(image, f"amount of cars: {len(center_points_cur_frame)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 255, 20), 2)

                if count <= 2:
                    for pt, cls in center_points_cur_frame:
                        for pt2, cls2 in center_points_prev_frame:

                            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                            
                            new_pt, new_pt2 = getRealcoords(pt), getRealcoords(pt2)
                            real_distance = math.hypot(new_pt2[0] - new_pt[0], new_pt2[1] - new_pt[1])

                            if distance < default_distance:
                                speed_t = (real_distance / 10000) / hours_perframe
                                tracking_objects[ids[0]] = pt, speed_t, cls
                                ids.remove(ids[0])

                else:
                    tracking_obejcts_copy = tracking_objects.copy()
                    center_points_cur_frame_copy = center_points_cur_frame.copy()
                    for object_id, (pt2, speed, cls2) in tracking_obejcts_copy.items():
                        object_exists = False
                        for pt, cls in center_points_cur_frame_copy:
                            
                            distance = math.hypot((pt2[0] - pt[0])*0.7, (pt2[1] - pt[1])*1.4)# Взято с потолка

                            new_pt, new_pt2 = getRealcoords(pt), getRealcoords(pt2)
                            real_distance = math.hypot(new_pt2[0] - new_pt[0], new_pt2[1] - new_pt[1])
                            
                            # Update IDs positionr
                            if distance < default_distance:
                                speed_t = (real_distance / 10000) / hours_perframe
                                tracking_objects[object_id] = pt, speed_t, cls
                                object_exists = True
                                if (pt, cls) in center_points_cur_frame:
                                    center_points_cur_frame.remove((pt, cls))
                                continue

                        # Remove id lost
                        if not object_exists:
                            tracking_objects.pop(object_id)
                            ids.append(object_id)

                    # Add new IDs found
                    for pt, cls in center_points_cur_frame:
                        tracking_objects[ids[0]] = pt, 0, cls
                        ids.remove(ids[0])

                for object_id, (pt, speed, cls) in tracking_objects.items():
                    image = cv2.circle(image, pt, 5, (0, 0, 255), -1)#object center
                    gps_coord = getRealcoords((pt[0], pt[1]), True)#GPS
                    image = cv2.putText(image, f"{str(object_id)} s: {'%.2f'%speed} km/h", (pt[0] - 12, pt[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                    lane_n = None
                    for k, traffic_lane in enumerate(traffic_lanes_path):
                        if traffic_lane.contains_point((pt[0], pt[1])): 
                            lane_n = k
                    if getResult: df_results.loc[len(df_results.index)] = [count, object_id, pt[0], pt[1], float(gps_coord[0]), float(gps_coord[1]), cls, speed, lane_n]

                image = cv2.resize(image, (x_resized,y_resized), interpolation = cv2.INTER_AREA)
                if video_rec: writer.write(image)
                cv2.imshow('frame', image)
                cv2.namedWindow('frame')
                cv2.setMouseCallback('frame', onMouse)

                center_points_prev_frame = center_points_cur_frame.copy()


                if cv2.waitKey(1) == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    if getResult: df_results.to_csv(os.path.join(CURR_DIR, 'out.csv'), index=False)
                    if video_rec: writer.close()
                    return
                
            else:
                break
                
        if online:
            del_file = local_file
            local_files.pop(0)
            local_files = download_files(local_files)


main()


"""
расстояние по y = 24.4 м
расстояние по x = 29.4 м

POLYGONE INTERPOLATION
                (0,244)--------(294,244)
                    |               |
Polygone ->         |               |
                    |               |
                  (0,0)--------(294, 0)
"""