import numpy as np
import math
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import cv2
import os


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       print('x = %d, y = %d'%(x, y))


def zoom_at(img, zoom=1, angle=0, coord=None):
    
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    
    rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result


pol1 = (582/1280, 271/720)
pol2 = (371/1280, 520/720)
pol3 = (945/1280, 710/720)
pol4 = (994/1280, 317/720)



def distort():

    k1, k2, p1, p2, k3, c_x, c_y, f_x, f_y = [-0.482867598990357,
                                              0.17660934951820867,
                                              -0.0069868318800345094,
                                              -0.0014074469204048575,
                                              -0.022916394527145435,
                                              1107.4142188343726,
                                              750.2347085083156,
                                              1559.2275934782917,
                                              1151.919978031152]

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
    img = zoom_at(img, 1.3, coord=(1093, 642))

    polygon = np.array([(pol1[0] * x_res, pol1[1] * y_res), 
                        (pol2[0] * x_res, pol2[1] * y_res),
                        (pol3[0] * x_res, pol3[1] * y_res),
                        (pol4[0] * x_res, pol4[1] * y_res)], dtype=int)#points for polygone in the center
    img = cv2.polylines(img, [polygon.reshape((-1, 1, 2))], True, (255, 4, 0), 2)
    
    img = cv2.resize(img, (1280,720), interpolation = cv2.INTER_AREA)
    cv2.imshow('image', img)
    cv2.imwrite(os.path.join(CURR_DIR, "new_image2.png"), img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', onMouse)
    cv2.waitKey(0)


distort()
cv2.destroyAllWindows()