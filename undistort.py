import numpy as np
import math
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import cv2
import os


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       print('x = %d, y = %d'%(x, y))


def getCurve(pts):
    coeffs = np.polyfit(pts[:,1], pts[:,0], 2)
    poly = np.poly1d(coeffs)
    yarr = np.arange(pts[0][1], pts[2][1])
    xarr = poly(yarr)
    parab_pts = np.array([xarr, yarr],dtype=np.int32).T
    return parab_pts


def getLine(point1, point2, curve_points):
    poly_x = point1[0], point2[0]
    poly_y = point1[1], point2[1]
    poly = np.poly1d(np.polyfit(poly_x, poly_y, 1))
    xarr = curve_points.T[0]
    yarr = poly(xarr).astype(int)
    line_pts = np.array([xarr, yarr],dtype=np.int32).T
    return line_pts
    
def getPoints(contours, *points):
    """
    Parsing 'contour array' to get point(s)
    """
    pointsRet = []
    for p in points:
        pointsRet.append( np.squeeze(contours[p][len(contours[p])//2]))
    return pointsRet



def getHsvColor(rgb):
    return np.squeeze(cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_BGR2HSV))
    

def distort(params):
    show_frame = True
    k1, k2, p1, p2, k3, c_x, c_y, f_x, f_y = params

    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(CURR_DIR, "image.png")
    img = cv2.imread(file)

    camera_matrix = np.array([[f_x, 0.00000000e+00, c_x],
                            [0.00000000e+00, f_y, c_y],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_coefs = np.array([k1, k2, p1, p2, k3])


    #Setting an anchor points
    pts_hor = np.array([[1, 405],# line_start_x, line_start_y
                        [596, 491],# clicked_x , clicked_y
                        [1915, 734]], np.int32)# line_end_x, line_end_y
    
    pts_ver = np.array([[1558, 379],# line_start_x, line_start_y
                        [1551, 658],# clicked_x , clicked_y
                        [1477, 1078]], np.int32)# line_end_x, line_end_y
    
    parab_pts_hor = getCurve(pts_hor)
    parab_pts_ver = getCurve(pts_ver)
    img = cv2.polylines(img, [parab_pts_hor], False, (0,255,255), 2)#draw a curve
    img = cv2.polylines(img, [parab_pts_ver], False, (0,100,255), 2)#draw a curve

    
    for i in range(3):
        #DRAW HORIZONTAL POINTS
        img = cv2.circle(img, (pts_hor[i][0], pts_hor[i][1]), 0, (0, 0, 255), 5)
        #DRAW VERTICAL POINTS
        img = cv2.circle(img, (pts_ver[i][0], pts_ver[i][1]), 0, (0, 0, 250), 5)


    #UNDISTORT BLOCK
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    img = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # FILTERING BLOCK
    #MASKS
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masked_p = cv2.inRange(hsv, (0, 255, 255), (0, 255, 255))
    masked_l = cv2.inRange(hsv, (20, 255, 255), (255, 255, 255))
    masked_p_ver = cv2.inRange(hsv, (0, 255, 250), (0, 255, 250))
    masked_l_ver = cv2.inRange(hsv, (12, 255, 255), (12, 255, 255))

    #CONTOURS BY MASKS
    contours_p, hierarchy_p = cv2.findContours(masked_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_p_ver, hierarchy_p_ver = cv2.findContours(masked_p_ver, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_l, hierarchy_l = cv2.findContours(masked_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_l_ver, hierarchy_l_ver = cv2.findContours(masked_l_ver, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours_p)!=3 or len(contours_p_ver)!=3: return 10000


    #POINTS FOR VERTICAL AND HORIZONTAL LINES
    point1_hor, point2_hor = getPoints(contours_p, 0, 2)
    point1_ver, point2_ver = getPoints(contours_p_ver, 0, 2)

    #POINTS FOR CURVES
    curve_points_hor = np.squeeze(np.concatenate(contours_l))
    curve_points_ver = np.squeeze(np.concatenate(contours_l_ver))

    gotLine_hor = getLine(point1_hor, point2_hor, curve_points_hor)
    gotLine_ver = getLine(point1_ver, point2_ver, curve_points_ver)
    img = cv2.polylines(img, [gotLine_hor], False, (255,0,0), 2)
    img = cv2.polylines(img, [gotLine_ver], False, (255,0,0), 2)


    mse_metric_hor = mean_squared_error(gotLine_hor.T[1], curve_points_hor.T[1])
    mse_metric_ver = mean_squared_error(gotLine_ver.T[1], curve_points_ver.T[1])

    myMetric = mse_metric_hor# + mse_metric_ver
    print(myMetric)

    if show_frame:
        # img = masked_angle_p_hor
        img = cv2.resize(img, (1280,720), interpolation = cv2.INTER_AREA)
        cv2.imshow('image', img)
        cv2.imwrite(os.path.join(CURR_DIR, "new_image2.png"), img)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', onMouse)
        cv2.waitKey(0)

    return myMetric



# OPTIMIZATION
# gues [-0.49181345,  0.25848255, -0.01067125, -0.00127517, -0.01900726, 9.40592038e+02, 5.96848905e+02, 1.26125746e+03, 1.21705719e+03]
# best without euclid equality [-4.42311140e-01, 2.13923672e-01, -8.25983108e-03, -1.44481839e-03, -2.06240076e-02, 1.01586176e+03, 7.14619454e+02, 1.51359801e+03, 1.11939455e+03]
#[-4.79778127e-01,  1.62865231e-01, -7.81805038e-03, -1.24619370e-03, -2.02551170e-02,  1.31145492e+03,  7.65014463e+02,  1.62265199e+03,  1.13263572e+03]
#-0.482867598990357,0.17660934951820867,-0.0069868318800345094,-0.0014074469204048575,-0.022916394527145435,1107.4142188343726,750.2347085083156,1559.2275934782917,1151.919978031152
#-0.4857928775445315,0.17392051187170604,-0.007052080661979939,-0.0014139588280523154,-0.02289154826235011,1109.2909568797954,746.8801576563435,1573.0563145555982,1155.2820182964092

#TOP1 -0.44122265390547605,0.21375771203025706,-0.008192271315560882,-0.00143372092373238,-0.020766579977952507,1028.4401269037626,706.6902157673671,1551.0929328383193,1138.611383662474

# x0 = [-4.42311140e-01, 2.13923672e-01, -8.25983108e-03, -1.44481839e-03, -2.06240076e-02, 1.01586176e+03, 7.14619454e+02, 1.51359801e+03, 1.11939455e+03]
# out = minimize(distort, x0, method='Nelder-Mead')
# print(*out.x, sep=",")

distort([-0.44122265390547605,0.21375771203025706,-0.008192271315560882,-0.00143372092373238,-0.020766579977952507,1028.4401269037626,706.6902157673671,1551.0929328383193,1138.611383662474])
cv2.destroyAllWindows()
