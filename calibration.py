import math
import os

import cv2
import numpy as np
import json
import glob

pictures = glob.glob('LV1\*.jpg')
print(f"Total pictures found: {len(pictures)}")

max_images = 5
board_width = 8
board_height = 6
square_length = 36

corners = []
image_points = []
board_size = (board_width, board_height)

successes = 0

found = False

for p in pictures:
    if successes >= max_images:
        pts = image_points[-1].reshape(board_height, board_width, 2)
        dx = np.diff(pts[:, :, 0], axis=1)
        avg_square_px = np.mean(np.abs(dx))
        break

    img = cv2.imread(p)

    if img is None:
        print("Failed to load:", p)
        continue

    img_clone = img.copy()
    img_gray = cv2.cvtColor(img_clone, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(img_gray, board_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if found:
        print("Chessboard found in:", p)
        corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(img_clone, board_size, corners2, found)
        image_points.append(corners2.reshape(-1,2))

        successes += 1

    else:
        print("Chessboard NOT found in:", p)

    cv2.imshow("Calibration", img_clone)
    cv2.waitKey(0)

cv2.destroyAllWindows()


if successes == max_images:
    print('Calibrating...')
    total_avg_error  = 0

    object_points = []

    for i in range(board_size[1]):
        for j in range(board_size[0]):
            object_points.append(np.array([j*square_length, i*square_length, 0]))

    object_points = np.array([object_points] * len(image_points), dtype=np.float32)#.reshape(1, -1)

    print(object_points)
    print(image_points)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_gray.shape[::-1], None, None)

    print('Re-projection error reported by calibrateCamera: ', rms)

    ok = cv2.checkRange(camera_matrix) and cv2.checkRange(dist_coeffs)

    if ok:
        print('Calibration succeeded')
    else:
        print('Calibration failed')

    out_dict = {'camera_matrix': camera_matrix.tolist(), 'dist_coeffs': dist_coeffs.tolist()}

    with open('camera_params.json', 'w') as f:
        json.dump(out_dict, f)

# --------------Second part------------------
# Loading and undistorting an image using the saved parameters
pictures_object = glob.glob('LV1_objects\*.jpg')
with open('camera_params.json', 'r') as f:
    params = json.load(f)

camera_matrix = np.array(params['camera_matrix'])
dist_coeffs = np.array(params['dist_coeffs'])

first_img = pictures_object[0]
img = cv2.imread(first_img)
h_first, w_first = img.shape[:2]

new_matrix_first, roi_first = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w_first, h_first), 1, (w_first, h_first))
undistorted_first = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_matrix_first)

# User selects the object of interest in the first undistorted image
roi = cv2.selectROI("Mark the object", undistorted_first, fromCenter=False) # ENTER or SPACE to confirm, c to cancel
x, y, w_roi, h_roi = roi
object_model = undistorted_first[y:y+h_roi, x:x+w_roi]

# Showing SIFT keypoints on the First marked image
sift = cv2.SIFT_create()
kp_first, des_first = sift.detectAndCompute(object_model, None)

cv2.imshow("Object Model", object_model)
cv2.imshow("SIFT Keypoints on First Image", cv2.drawKeypoints(object_model, kp_first, None))
cv2.waitKey(0)

# Funtion for clicking coordinates on milimeter paper
points = []
def click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x,y])
        print("Clicked:",x,y)

cv2.imshow("image", undistorted_first)
cv2.setMouseCallback("image", click)
cv2.waitKey(0)

image_pts = np.array(points, dtype=np.float32)

y = 270
x = 180
world_pts = np.array([
    [0,0],
    [0,y],
    [x,y],
    [x,0]
], dtype=np.float32)

H, _ = cv2.findHomography(image_pts, world_pts)

for p in pictures_object[1:]:
    print("Processing:", p)
    img = cv2.imread(p)
    h, w = img.shape[:2]

    # Undistorting the rest of the images 
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    # Detecting SIFT keypoints and descriptors 
    kp_sec, des_sec = sift.detectAndCompute(undistorted_img, None)

    # Matching keypoints between the first and second image using BFMatcher 
    bf =  cv2.BFMatcher()
    matches = bf.knnMatch(des_first, des_sec, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Estimating the homography between the first and second image and drawing the matched keypoints and lines between them
    src_pts = np.float32([ kp_first[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_sec[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h_m, w_m = object_model.shape[:2]
    pts = np.float32([[0, 0], [0, h_m - 1], [w_m - 1, h_m - 1], [w_m - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts,M)
    dst += (w_m, 0)
    dst_real = cv2.perspectiveTransform(pts,M)

    img_matches = cv2.drawMatches(object_model, kp_first, undistorted_img, kp_sec, good_matches, None, matchColor=(255,0,0), singlePointColor=(255,0,0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches = cv2.polylines(img_matches, [np.int32(dst)], True, (255,255,255),1, cv2.LINE_AA)

    # Centroid
    if len(good_matches) > 0:
        centroid = np.mean(dst, axis=0)
        cv2.circle(img_matches, (int(centroid[0][0]), int(centroid[0][1])), 5, (0,255,0), -1)
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)

    # Coordinates and orientation
    pts = dst_real.reshape(4,2)
    cx = np.mean(pts[:,0])
    cy = np.mean(pts[:,1])
    pixel = np.array([[[cx,cy]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pixel, H)

    x_mm = world[0][0][0]
    y_mm = world[0][0][1]

    vec = pts[0] - pts[1] # Vector along the X-axis of the object
    angle_rad = math.atan2(vec[1], vec[0])
    angle_deg = math.degrees(angle_rad) 

    # Drawing the orientation arrow
    start_point = (int(pts[0][0]), int(pts[0][1]))
    end_point = (int(pts[1][0]), int(pts[1][1]))
    cv2.line(img_matches, start_point, end_point, (0, 0, 255), 3)
    cv2.arrowedLine(img_matches, start_point, end_point, (0, 0, 255), 3, tipLength=0.3)
    cv2.imshow("Orientation", img_matches)
    cv2.waitKey(0)

    print("Object position on paper:")
    print("X =", x_mm, "mm")
    print("Y =", y_mm, "mm")
    print("Orientation:", angle_deg, "degrees")

cv2.destroyAllWindows()


