import numpy as np
import math
import matplotlib.path as mplPath
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import cv2
import os


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       print('x = %d, y = %d'%(x/1280*1920, y/720*1080))


x_res, y_res = (1920, 1080)# RESOLUTION
x_resized, y_resized = (1280, 720) # NEW RESIZED RESOLUTION

#devided values got from new resolution (thant's why I'm making a devision)
pol1 = (554/x_resized, 237/y_resized)
pol2 = (343/x_resized, 490/y_resized)
pol3 = (934/x_resized, 693/y_resized)
pol4 = (978/x_resized, 278/y_resized)

#points for above line
pol5 = (650/x_resized*x_res, 203/y_resized*y_res)
pol6 = (887/x_resized*x_res, 220/y_resized*y_res)
line5_6 = np.array([pol5, pol6], dtype=int)

pol1_2 = (461/x_resized, 349/y_resized)
pol2_2 = (583/x_resized, 572/y_resized)
pol3_2 = (961/x_resized, 438/y_resized)
pol4_2 = (727/x_resized, 254/y_resized)


def line(img, p1:tuple, p2:tuple, left=True):
    """
    img - cv2 image
    p1 - start point
    p2 - end point
    """
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    colors = [
        (10, 10, 242),
        (10, 211, 242),
        (0, 247, 87),
        (243, 247, 0),
        (247, 103, 0),
        (247, 0, 169),
        (255, 255, 255)
    ]

    traffic_lanes = np.array([[(610, 674), (552, 746), (1348, 1004), (1369, 870)],
                            [(673, 595), (610, 674), (1369, 870), (1390, 735)],
                            [(726, 528), (673, 595), (1390, 735), (1404, 645)],
                            [(775, 467), (726, 528), (1404, 645), (1417, 562)],
                            [(817, 414), (775, 467), (1417, 562), (1429, 485)],
                            [(859, 362), (817, 414), (1429, 485), (1439, 421)],
                            [(961, 304), (930, 360), (1335, 400), (1339, 340)]],dtype=int)
    

    x = (np.arange(p1[0], p2[0],dtype=int))
    # print(x)
    if left:
        points = [x[0], 610, 673, 726, 775, 817, x[-1]]
    else:
        points = [x[0], 1369, 1390, 1404, 1417, 1429, x[-1]]
    
    poly = np.poly1d(np.polyfit([p1[0], p2[0]], [p1[1],p2[1]], 1))

    for i in range(len(traffic_lanes)):
        # img = cv2.line(img, (points[i], int(poly(points[i]))), (points[i+1], int(poly(points[i+1]))), colors[i], 3)
        # print((points[i], int(poly(points[i]))), (points[i+1], int(poly(points[i+1]))))
        img = cv2.polylines(img, [traffic_lanes[i].reshape((-1, 1, 2))], True, colors[i], 2)
        img = cv2.putText(img, str(i), (traffic_lanes[i][1][0] + 15, traffic_lanes[i][1][1]), cv2.LINE_4, 1, colors[i], 3)
    return img



def distort():

    k1, k2, p1, p2, k3, c_x, c_y, f_x, f_y = [-0.44122265390547605,0.21375771203025706,-0.008192271315560882,-0.00143372092373238,-0.020766579977952507,1028.4401269037626,706.6902157673671,1551.0929328383193,1138.611383662474]

    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(CURR_DIR, "image.png")
    img = cv2.imread(file)
    y_res, x_res = img.shape[0], img.shape[1]#resolution
    

    camera_matrix = np.array([[f_x, 0.00000000e+00, c_x],
                            [0.00000000e+00, f_y, c_y],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_coefs = np.array([k1, k2, p1, p2, k3])

    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    img = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    polygon = np.array([(pol1[0] * x_res, pol1[1] * y_res), 
                        (pol2[0] * x_res, pol2[1] * y_res),
                        (pol3[0] * x_res, pol3[1] * y_res),
                        (pol4[0] * x_res, pol4[1] * y_res)], dtype=int)#points for polygone in the center
    
    polygon2 = np.array([(pol1[0] * x_res, pol1[1] * y_res), 
                        (pol1_2[0] * x_res, pol1_2[1] * y_res),
                        (pol2[0] * x_res, pol2[1] * y_res),
                        (pol2_2[0] * x_res, pol2_2[1] * y_res),
                        (pol3[0] * x_res, pol3[1] * y_res),
                        (pol3_2[0] * x_res, pol3_2[1] * y_res),
                        (pol4[0] * x_res, pol4[1] * y_res), 
                        (pol4_2[0] * x_res, pol4_2[1] * y_res)], dtype=int)#points for polygone in the center
    
    
    img = cv2.polylines(img, [polygon.reshape((-1, 1, 2))], True, (255, 4, 0), 2)

    img = cv2.line(img, line5_6[0], line5_6[1], (255, 4, 0), 2)

    img = line(img, (573 / x_resized * x_res, 241 / y_resized * y_res), (368 / x_resized * x_res, 498 / y_resized * y_res))
    img = line(img, (1440, 415), (1348, 1005), left=False)

    for k, point in enumerate(polygon2):
        img = cv2.circle(img, point, 5, (0, 0, 255), 5)
        img = cv2.putText(img, str(k+1), (point[0] + 10, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
    
    img = cv2.resize(img, (x_resized, y_resized), interpolation = cv2.INTER_AREA)
    cv2.imshow('image', img)
    cv2.imwrite(os.path.join(CURR_DIR, "new_image2.png"), img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', onMouse)
    cv2.waitKey(0)


distort()
cv2.destroyAllWindows()