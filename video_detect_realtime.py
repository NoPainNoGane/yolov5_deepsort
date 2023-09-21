import numpy as np
import pandas as pd
import os
import torch
import cv2
import math
import m3u8
import urllib.request
import matplotlib.path as mplPath

import tqdm

from vidgear.gears import WriteGear

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

online = False

model_path = os.path.join(CURR_DIR, r"runs\train\yolov5s_ufa2\weights\best.pt")
video_path = os.path.join(CURR_DIR, r"data\videos\burger_new_6.mp4")

playlist = "http://136.169.226.59/1-4/tracks-v1/mono.m3u8?token=232d35c22d4e40d8ba47a8a2c35d2612"
videoLink = os.path.dirname(playlist) + '/'

#MODEL PARAMETERS
model = torch.hub.load('Ultralytics/yolov5', 'custom', model_path, trust_repo=True).eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.conf = 0.45
model.iou = 0.5

#Result Dataframe
getResult = True
df_results = pd.DataFrame(columns=['frame', 'id', 'x', 'y', 'lat', 'long', 'class', 'speed', 'lane', 'azimuth'])
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

traffic_lanes = np.array([[(514, 735), (576, 660), (1419, 869), (1401, 1038)],
                        [(576, 660), (636, 588), (1432, 746), (1419, 869)],
                        [(636, 588), (693, 520), (1443, 643), (1432, 746)],
                        [(693, 520), (738, 466), (1451, 567), (1443, 643)],
                        [(738, 466), (778, 418), (1459, 492), (1451, 567)],
                        [(778, 418), (830, 356), (1467, 417), (1459, 492)],
                        [(975, 304), (830, 356), (1467, 417), (1330, 330)]],dtype=int)

traffic_lanes_path = [mplPath.Path(lane) for lane in traffic_lanes]

ids = list(range(512))
frames_persec = 25
hours_perframe = 1 / 60 / 60 / frames_persec
default_distance = 25#euclid metric

from geographiclib.geodesic import Geodesic

def get_bearing(lat1, long1, lat2, long2 ):
    brng = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']
    return brng

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def calc_bearing(llat1, llong1, llat2, llong2):
    #pi - число pi, rad - радиус сферы (Земли)
    rad = 6372795
    
    #координаты двух точек
    # llat1 = 77.1539
    # llong1 = -120.398
    
    # llat2 = 77.1804
    # llong2 = 129.55
    
    #в радианах
    lat1 = llat1*math.pi/180.
    lat2 = llat2*math.pi/180.
    long1 = llong1*math.pi/180.
    long2 = llong2*math.pi/180.
    
    #косинусы и синусы широт и разницы долгот
    cl1 = math.cos(lat1)
    cl2 = math.cos(lat2)
    sl1 = math.sin(lat1)
    sl2 = math.sin(lat2)
    delta = long2 - long1
    cdelta = math.cos(delta)
    sdelta = math.sin(delta)
    
    #вычисления длины большого круга
    y = math.sqrt(math.pow(cl2*sdelta,2)+math.pow(cl1*sl2-sl1*cl2*cdelta,2))
    x = sl1*sl2+cl1*cl2*cdelta
    ad = math.atan2(y,x)
    dist = ad*rad
    
    #вычисление начального азимута
    x = (cl1*sl2) - (sl1*cl2*cdelta)
    y = sdelta*cl2
    z = math.degrees(math.atan(-y/x))
    
    if (x < 0):
        z = z+180.
    
    z2 = (z+180.) % 360. - 180.
    z2 = - math.radians(z2)
    anglerad2 = z2 - ((2*math.pi)*math.floor((z2/(2*math.pi))) )
    angledeg = (anglerad2*180.)/math.pi
    
    return angledeg

def get_bearing1(lat1,lon1,lat2,lon2):
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon)
    brng = np.rad2deg(math.atan2(y, x))
    if brng < 0: brng+= 180
    return brng

def get_bearing2(xLat, xLng, yLat, yLng):
    dLat = yLat - xLat
    dLng = yLng - xLng
    angle = (math.atan2(dLng, dLat))
    if angle < 0: angle += 360
    
def get_bearing3(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)

    return brng    

def distance_bearing(homeLatitude, homeLongitude, destinationLatitude, destinationLongitude):
    R = 6371e3
    """
    Simple function which returns the distance and bearing between two geographic location

    Inputs:
        1.  homeLatitude            -   Latitude of home location
        2.  homeLongitude           -   Longitude of home location
        3.  destinationLatitude     -   Latitude of Destination
        4.  destinationLongitude    -   Longitude of Destination

    Outputs:
        1. [Distance, Bearing]      -   Distance (in metres) and Bearing angle (in degrees)
                                        between home and destination

    Source:
        https://github.com/TechnicalVillager/distance-bearing-calculation
    """

    rlat1   =   homeLatitude * (math.pi/180) 
    rlat2   =   destinationLatitude * (math.pi/180) 
    rlon1   =   homeLongitude * (math.pi/180) 
    rlon2   =   destinationLongitude * (math.pi/180) 
    dlat    =   (destinationLatitude - homeLatitude) * (math.pi/180)
    dlon    =   (destinationLongitude - homeLongitude) * (math.pi/180)

    # Haversine formula to find distance
    a = (math.sin(dlat/2) * math.sin(dlat/2)) + (math.cos(rlat1) * math.cos(rlat2) * (math.sin(dlon/2) * math.sin(dlon/2)))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # Distance in metres
    distance = R * c

    # Formula for bearing
    y = math.sin(rlon2 - rlon1) * math.cos(rlat2)
    x = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(rlon2 - rlon1)
    
    # Bearing in radians
    bearing = math.atan2(y, x)
    bearingDegrees = bearing * (180/math.pi)
    out = [distance, bearingDegrees]

    return out

def get_bearing4(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    Base = 6371 * c


    Bearing = math.atan2(math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1), math.sin(lon2-lon1)*math.cos(lat2))

    Bearing = math.degrees(Bearing)
    
    return Bearing

def calc_angle(ax, ay, bx, by):
    ma = math.sqrt(ax * ax + ay * ay)
    mb = math.sqrt(bx * bx + by * by)
    sc = ax * bx + ay * by
    res = np.degrees(math.acos(sc / ma / mb))
    return res



import pyproj
geodesic = pyproj.Geod(ellps='WGS84')



import time # 14.09.23
from deep_sort_realtime.deepsort_tracker import DeepSort # 14.09.23
import sort # 14.09.23
import cvzone

object_tracker = DeepSort(max_age=5,
                          n_init=3,
                          nms_max_overlap=1.0,
                          max_cosine_distance=0.3,
                          nn_budget=None,
                          override_track_class=None,
                          embedder='mobilenet',
                          half=True,
                          bgr=True,
                          embedder_gpu=True,
                          embedder_model_name=None,
                          embedder_wts=None,
                          polygon=False,
                          today=None) # 14.09.23

# ------------------------SORT--------------------------------
# object_tracker = sort.Sort(max_age=50, min_hits=2, iou_threshold=0.5)


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
    
    classNames = ['bus', 'car', 'pedestrian', 'truck', 'van']
    
    while True:
        if online:
            local_file = local_files[0]
            cap = cv2.VideoCapture(local_file)
            if del_file:
                os.remove(del_file)
        else:
            cap = cv2.VideoCapture(video_path)
        
        file_frames = 0
        
        while cap.isOpened(): # 14.09.23
            success, frame = cap.read()
            file_frames += 1
            if file_frames == 1: continue
            
            if success == True:
                #reduction of distortion
                frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h,  w = frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
                frame = cv2.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                count += 1
                
                results = model(frame)
                # ------------------------SORT--------------------------------
                # detections = np.empty((0, 5))
                detections = []
                nameList = []
                
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
                        
                        # image = cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]), color_dict[tup[5]], 8)# object rectangle
                        # image = cv2.rectangle(image, (dist_x[0], dist_y[0]), (dist_x[1], dist_y[1]), (255,255,255), 1)# euclid metric rectangle
                        # image = cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)# center of object
                        image = cv2.putText(image, tup[6], (x[0], y[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # label of object
                        image = cv2.putText(image, str(gps_coord[0]) +" "+ str(gps_coord[1]), (x[0] - 30, y[0] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)# GPS label of object
                        center_points_cur_frame.append(((cx, cy), tup[6]))
                        
                        currentArray = np.array([x[0], y[0], x[1], y[1], tup[4]])
                        nameList.append(tup[6])
                        # ------------------------SORT--------------------------------
                        # detections = np.vstack((detections, currentArray))
                        detections.append(([x[0], y[0], abs(x[1]-x[0]), abs(y[1]-y[0])], tup[4], tup[6]))
                        
                # ------------------------SORT--------------------------------
                # resultsTracker = object_tracker.update(detections)
                resultsTracker = object_tracker.update_tracks(detections, frame=frame)
                nameCount = 0
                for track in resultsTracker:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    # print(track.get_det_class)
                    bbox = ltrb
                    a = track
                    image = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
                    # image = cv2.putText(image, str(track_id), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1) # label of object
                    
                    image = cv2.putText(image, track.get_det_class(), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # label of object
                    # image = cv2.putText(image, str(gps_coord[0]) +" "+ str(gps_coord[1]), (x[0] - 30, y[0] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)# GPS label of object
                    nameCount += 1
                
                # ------------------------SORT--------------------------------
                # for result in resultsTracker: 
                #     x1, y1, x2, y2, id = result
                #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                #     cx = int(x1 - (x1 - x2) / 2)
                #     cy = int(y1 + (y2 - y1) / 2)
                    
                #     gps_coord = getRealcoords((cx, cy), True)
                    
                #     # print(result)
                #     image = cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
                #     image = cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)# center of object
                #     image = cv2.putText(image, str(id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1) # label of object
                #     image = cv2.putText(image, str(gps_coord[0]) +" "+ str(gps_coord[1]), (x1 - 30, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)# GPS label of object
                #     # center_points_cur_frame.append(((cx, cy), nameList))
                    
                
                
                sub_img = image[0:437, 0:470]
                white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
                res = cv2.addWeighted(sub_img, 0.4, white_rect, 0.5, 1.0)
                image[0:437, 0:470] = res

                image = cv2.polylines(image, [polygon.reshape((-1, 1, 2))], True, (255, 4, 0), 2)

                
                for k, lane in enumerate(traffic_lanes):
                    image = cv2.polylines(image, [lane.reshape((-1, 1, 2))], True, colors_traffic[k], 2)
                    image = cv2.putText(image, str(k), (lane[0][0] + 15, lane[0][1]), cv2.LINE_4, 1, colors_traffic[k], 2)

                image = cv2.putText(image, f"amount of cars: {len(center_points_cur_frame)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 255, 20), 2)
                
                if count <= 2:
                    for pt, cls in center_points_cur_frame:
                        for pt2, cls2 in center_points_prev_frame:

                            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                            
                            new_pt, new_pt2 = getRealcoords(pt), getRealcoords(pt2)
                            real_distance = math.hypot(new_pt2[0] - new_pt[0], new_pt2[1] - new_pt[1])

                            az_pt, az_pt2 = getRealcoords(pt, True), getRealcoords(pt2, True) # 21.09
                            
                            if distance < default_distance:
                                azimuth = get_bearing(az_pt[0], az_pt[1], 
                                            az_pt2[0], az_pt2[1]) # 21.09
                                speed_t = (real_distance / 10000) / hours_perframe
                                tracking_objects[ids[0]] = pt, speed_t, cls, azimuth
                                ids.remove(ids[0])

                else:
                    tracking_obejcts_copy = tracking_objects.copy()
                    center_points_cur_frame_copy = center_points_cur_frame.copy()
                    for object_id, (pt2, speed, cls2, azimuth) in tracking_obejcts_copy.items():
                        object_exists = False
                        for pt, cls in center_points_cur_frame_copy:
                            
                            distance = math.hypot((pt2[0] - pt[0])*0.7, (pt2[1] - pt[1])*1.4)# Взято с потолка

                            new_pt, new_pt2 = getRealcoords(pt), getRealcoords(pt2)
                            real_distance = math.hypot(new_pt2[0] - new_pt[0], new_pt2[1] - new_pt[1])
                            
                            az_pt, az_pt2 = getRealcoords(pt, True), getRealcoords(pt2, True) # 21.09
                            
                            # Update IDs positionr
                            if distance < default_distance:
                                azimuth = get_bearing(az_pt[0], az_pt[1], 
                                            az_pt2[0], az_pt2[1]) # 21.09
                                speed_t = (real_distance / 10000) / hours_perframe
                                tracking_objects[object_id] = pt, speed_t, cls, azimuth
                                # print(az_pt[0], az_pt[1], az_pt2[0], az_pt2[1], azimuth) # 21.09
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
                        tracking_objects[ids[0]] = pt, 0, cls, 0
                        ids.remove(ids[0])


                lane_count = {i:0 for i in range(7)}
                gps = []
                az_count = 0
                for object_id, (pt, speed, cls, azimuth) in tracking_objects.items():
                    image = cv2.circle(image, pt, 5, (0, 0, 255), -1)#object center
                    gps_coord = getRealcoords((pt[0], pt[1]), True)#GPS
                    gps.append(getRealcoords((pt[0], pt[1]), True))
                    if az_count > 0:
                        # fwd_azimuth,back_azimuth,distance = geodesic.inv(float(gps[az_count][0]), float(gps[az_count][1]), 
                        #                                                  float(gps[az_count - 1][0]), float(gps[az_count - 1][1]))
                        
                        # azimuth = get_bearing(float(gps[az_count][0]), float(gps[az_count][1]), 
                        #                       float(gps[az_count - 1][0]), float(gps[az_count - 1][1]))
                        # ax = float(gps[az_count][0]) - float(gps[az_count - 1][0])
                        # ay = float(gps[az_count][1]) - float(gps[az_count - 1][1])
                        # north_x, north_y = 54.725498 - 54.725357, 55.940651 - 55.940654
                        # angle = calc_angle(ax, ay, north_x, north_y)
                        
                        # azimuth = get_bearing(float(gps[az_count - 1][0]), float(gps[az_count - 1][1]), 
                        #                      float(gps[az_count][0]), float(gps[az_count][1]))
                        
                        # azimuth = calc_bearing(float(gps[az_count][0]), float(gps[az_count][1]), 
                        #                       float(gps[az_count - 1][0]), float(gps[az_count - 1][1]))
                        
                        # azimuth = calculate_initial_compass_bearing((float(gps[az_count][0]), float(gps[az_count][1])), (float(gps[az_count - 1][0]), float(gps[az_count - 1][1])))
                        # image = cv2.putText(image, f"{str(object_id)} s: {'%.2f'%speed} km/h  az: {azimuth}", (pt[0] - 12, pt[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        
                        image = cv2.putText(image, f"{str(object_id)} az: {azimuth}", (pt[0] - 12, pt[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        
                        # image = cv2.putText(image, f"{str(object_id)} az: {fwd_azimuth}", (pt[0] - 12, pt[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    az_count += 1
                    
                    

                    lane_n = None
                    for k, traffic_lane in enumerate(traffic_lanes_path):
                        if traffic_lane.contains_point((pt[0], pt[1])):
                            lane_n = k
                            lane_count[k] += 1
                    if getResult: df_results.loc[len(df_results.index)] = [count, object_id, pt[0], pt[1], float(gps_coord[0]), float(gps_coord[1]), cls, speed, lane_n, azimuth]

                
                
                
                for i in range(7):
                    image = cv2.putText(image, f"lane {i} count: " + str(lane_count[i]), (20, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors_traffic[i], 2)

                image = cv2.resize(image, (x_resized,y_resized), interpolation = cv2.INTER_AREA)
                if video_rec: writer.write(image)
                cv2.imshow('frame', image)
                cv2.namedWindow('frame')
                cv2.setMouseCallback('frame', onMouse)

                center_points_prev_frame = center_points_cur_frame.copy()
                
                # cv2.imshow('frame', frame)
                # cv2.waitKey(1)
                
                if cv2.waitKey(1) == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    if getResult: df_results.to_csv(os.path.join(CURR_DIR, 'out.csv'), index=False)
                    if video_rec: writer.close()
                    return
            
            
        
        
        
        
        
        
        # while cap.isOpened():
            
        #     ret, frame = cap.read()
            
        #     file_frames += 1
        #     if file_frames == 1: continue

        #     if ret == True:   
        #         #reduction of distortion
        #         frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         h,  w = frame.shape[:2]
        #         newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
        #         frame = cv2.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #         count += 1

        #         results = model(frame)
        #         df = results.pandas().xyxy[0]
        #         tups = list(df.itertuples(index=False))

        #         center_points_cur_frame = []
        #         image = frame.copy()
                
        #         for tup in tups:
        #             x = (int(tup[0]), int(tup[2]))
        #             y = (int(tup[1]), int(tup[3]))
        #             cx = int(x[0] - (x[0] - x[1]) / 2)
        #             cy = int(y[0] + (y[1] - y[0]) / 2)
                    
        #             dist_x = ( cx - default_distance, cx + default_distance )
        #             dist_y = ( cy + default_distance, cy - default_distance )

        #             if tup[5] != 2 and poly_path.contains_point((cx, cy)):

        #                 gps_coord = getRealcoords((cx, cy), True)

        #                 image = cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]), color_dict[tup[5]], 2)# object rectangle
        #                 image = cv2.rectangle(image, (dist_x[0], dist_y[0]), (dist_x[1], dist_y[1]), (255,255,255), 1)# euclid metric rectangle
        #                 image = cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)# center of object
        #                 image = cv2.putText(image, tup[6], (x[0], y[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # label of object
        #                 image = cv2.putText(image, str(gps_coord[0]) +" "+ str(gps_coord[1]), (x[0] - 30, y[0] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)# GPS label of object
        #                 center_points_cur_frame.append(((cx, cy), tup[6]))

        #         sub_img = image[0:437, 0:470]
        #         white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        #         res = cv2.addWeighted(sub_img, 0.4, white_rect, 0.5, 1.0)
        #         image[0:437, 0:470] = res

        #         image = cv2.polylines(image, [polygon.reshape((-1, 1, 2))], True, (255, 4, 0), 2)

                
        #         for k, lane in enumerate(traffic_lanes):
        #             image = cv2.polylines(image, [lane.reshape((-1, 1, 2))], True, colors_traffic[k], 2)
        #             image = cv2.putText(image, str(k), (lane[0][0] + 15, lane[0][1]), cv2.LINE_4, 1, colors_traffic[k], 2)

        #         image = cv2.putText(image, f"amount of cars: {len(center_points_cur_frame)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 255, 20), 2)

        #         if count <= 2:
        #             for pt, cls in center_points_cur_frame:
        #                 for pt2, cls2 in center_points_prev_frame:

        #                     distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                            
        #                     new_pt, new_pt2 = getRealcoords(pt), getRealcoords(pt2)
        #                     real_distance = math.hypot(new_pt2[0] - new_pt[0], new_pt2[1] - new_pt[1])

        #                     if distance < default_distance:
        #                         speed_t = (real_distance / 10000) / hours_perframe
        #                         tracking_objects[ids[0]] = pt, speed_t, cls
        #                         ids.remove(ids[0])

        #         else:
        #             tracking_obejcts_copy = tracking_objects.copy()
        #             center_points_cur_frame_copy = center_points_cur_frame.copy()
        #             for object_id, (pt2, speed, cls2) in tracking_obejcts_copy.items():
        #                 object_exists = False
        #                 for pt, cls in center_points_cur_frame_copy:
                            
        #                     distance = math.hypot((pt2[0] - pt[0])*0.7, (pt2[1] - pt[1])*1.4)# Взято с потолка

        #                     new_pt, new_pt2 = getRealcoords(pt), getRealcoords(pt2)
        #                     real_distance = math.hypot(new_pt2[0] - new_pt[0], new_pt2[1] - new_pt[1])
                            
        #                     # Update IDs positionr
        #                     if distance < default_distance:
        #                         speed_t = (real_distance / 10000) / hours_perframe
        #                         tracking_objects[object_id] = pt, speed_t, cls
        #                         object_exists = True
        #                         if (pt, cls) in center_points_cur_frame:
        #                             center_points_cur_frame.remove((pt, cls))
        #                         continue

        #                 # Remove id lost
        #                 if not object_exists:
        #                     tracking_objects.pop(object_id)
        #                     ids.append(object_id)

        #             # Add new IDs found
        #             for pt, cls in center_points_cur_frame:
        #                 tracking_objects[ids[0]] = pt, 0, cls
        #                 ids.remove(ids[0])


        #         lane_count = {i:0 for i in range(7)}

        #         for object_id, (pt, speed, cls) in tracking_objects.items():
        #             image = cv2.circle(image, pt, 5, (0, 0, 255), -1)#object center
        #             gps_coord = getRealcoords((pt[0], pt[1]), True)#GPS
        #             image = cv2.putText(image, f"{str(object_id)} s: {'%.2f'%speed} km/h", (pt[0] - 12, pt[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        #             lane_n = None
        #             for k, traffic_lane in enumerate(traffic_lanes_path):
        #                 if traffic_lane.contains_point((pt[0], pt[1])):
        #                     lane_n = k
        #                     lane_count[k] += 1
        #             if getResult: df_results.loc[len(df_results.index)] = [count, object_id, pt[0], pt[1], float(gps_coord[0]), float(gps_coord[1]), cls, speed, lane_n]

        #         for i in range(7):
        #             image = cv2.putText(image, f"lane {i} count: " + str(lane_count[i]), (20, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors_traffic[i], 2)

        #         image = cv2.resize(image, (x_resized,y_resized), interpolation = cv2.INTER_AREA)
        #         if video_rec: writer.write(image)
        #         cv2.imshow('frame', image)
        #         cv2.namedWindow('frame')
        #         cv2.setMouseCallback('frame', onMouse)

        #         center_points_prev_frame = center_points_cur_frame.copy()


        #         if cv2.waitKey(1) == ord('q'):
        #             cap.release()
        #             cv2.destroyAllWindows()
        #             if getResult: df_results.to_csv(os.path.join(CURR_DIR, 'out.csv'), index=False)
        #             if video_rec: writer.close()
        #             return
                
        #     else:
        #         break
                
        # if online:
        #     del_file = local_file
        #     local_files.pop(0)
        #     local_files = download_files(local_files)


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