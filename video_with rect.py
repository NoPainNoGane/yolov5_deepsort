import numpy as np
import math
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import cv2
import os


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       print('x = %d, y = %d'%(x, y))


pol1 = (554/1280, 237/720)
pol2 = (343/1280, 490/720)
pol3 = (934/1280, 693/720)
pol4 = (978/1280, 278/720)

pol1_2 = (461/1280, 349/720)
pol2_2 = (583/1280, 572/720)
pol3_2 = (961/1280, 438/720)
pol4_2 = (727/1280, 254/720)



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

    for k, point in enumerate(polygon2):
        img = cv2.circle(img, point, 5, (0, 0, 255), 5)
        img = cv2.putText(img, str(k+1), (point[0] + 10, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
    
    img = cv2.resize(img, (1280,720), interpolation = cv2.INTER_AREA)
    cv2.imshow('image', img)
    cv2.imwrite(os.path.join(CURR_DIR, "new_image2.png"), img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', onMouse)
    cv2.waitKey(0)


distort()
cv2.destroyAllWindows()