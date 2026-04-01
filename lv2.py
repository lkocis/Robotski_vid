import cv2
import numpy as np
import glob
import json
from convert_2d_points_to_3d_points import convert_2d_points_to_3d_points
from plot_3d_points import main as plot_3d_points
import open3d as o3d

# Load the images
pictures = glob.glob('LV2_object\*.jpg')
print(f"Total pictures found: {len(pictures)}")
cv2.imshow('First img', cv2.imread(pictures[0]))
cv2.imshow('Second img', cv2.imread(pictures[1]))
cv2.waitKey(0)
cv2.destroyAllWindows()

img1 = cv2.imread(pictures[0])
img2 = cv2.imread(pictures[1])

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for the first image
keypoints_first = []
descriptors_first = []
kp_first, des_first = sift.detectAndCompute(img1_gray, None)
keypoints_first.append(kp_first)
descriptors_first.append(des_first)

cv2.imshow("SIFT Keypoints on FirstImage", cv2.drawKeypoints(img1_gray, kp_first, None))

# Detect keypoints and compute descriptors for the second image
keypoints_second = []
descriptors_second = []

kp_sec, des_sec = sift.detectAndCompute(img2_gray, None)
keypoints_second.append(kp_sec)
descriptors_second.append(des_sec)

cv2.imshow("SIFT Keypoints on Second Image", cv2.drawKeypoints(img2_gray, kp_sec, None))
cv2.waitKey(0)

# Matching keypoints between the first image and the second image using BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors_first[0], descriptors_second[0], k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

img_matches = cv2.drawMatches(img1_gray, kp_first, img2_gray, kp_sec, good_matches, None, matchColor=(255,0,0), singlePointColor=(255,0,0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Good Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Estimating fundamental matrix using RANSAC
src_pts = np.float32([kp_first[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_sec[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

# Selecting inlier matches based on the mask
inlier_matches = []
for i, m in enumerate(good_matches):
    if mask[i] == 1:
        inlier_matches.append(m)
img_inlier_matches = cv2.drawMatches(img1_gray, kp_first, img2_gray, kp_sec, inlier_matches, None, matchColor=(0,255,0), singlePointColor=(0,255,0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Inlier Matches", img_inlier_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Essential matrix estimation
with open('camera_params.json', 'r') as f:
    params = json.load(f)
camera_matrix = np.array(params['camera_matrix'])
P = np.matrix(camera_matrix)
E = P.T * np.matrix(F) * P

# Estimate the 3D points from the 2D correspondences using the essential matrix
points_2d_L = [kp_first[m.queryIdx] for m in inlier_matches]
points_2d_R = [kp_sec[m.trainIdx] for m in inlier_matches]
coords_3d = convert_2d_points_to_3d_points(points_2d_L, points_2d_R, E, P)
with open('points_3d.json', 'w') as f:
    json.dump(coords_3d.tolist(), f)


# Plot the 3D points
plot_3d_points()

# Visualize 3D points with Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords_3d)
o3d.visualization.draw_geometries([pcd])