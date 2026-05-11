import numpy as np
import cv2
from read_kinect_pic import read_kinect_pic
import os
import pyrealsense2 as rs

def take_pictures(path):
    if not os.path.exists(path):
        os.makedirs(path)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    count = 0

    print("Camera started. Press SPACE to take a picture, ESC to exit.")

    try:
        while count < 10:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Show the color image
            cv2.imshow('Color Image', color_image)

            key = cv2.waitKey(1)
            if key == ord(' '):
                cv2.imwrite(os.path.join(path, f"sl-{count:05d}.bmp"), color_image)
                np.savetxt(os.path.join(path, f"sl-{count:05d}-D.txt"), depth_image, fmt='%d')
                print(f"Saved depth and color images for count {count}")
                count += 1

            elif key == 27:  # ESC key to exit
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def find_dominant_plane(folder_path):
    img_shape = (480, 640)

    # RANSAC parameters
    iterations = 500
    distance_threshold = 5
    num_planes = 3
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    for filename in os.listdir(folder_path):
        if filename.endswith("-D.txt"):
            depth_path = os.path.join(folder_path, filename)
            depth_image, point_3d_array, n_3d_points = read_kinect_pic(depth_path, img_shape)

            print(f"Processed {filename}:")
            print(f"Depth image shape: {depth_image.shape}")
            print(f"Number of 3D points: {n_3d_points}")

            points = np.array(point_3d_array)
            output_img = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

            print("Running RANSAC to find dominant plane...")
            # RANSAC to find dominant planes
            for i in range(num_planes):
                if len(points) < 3:
                    print("Not enough points to fit a plane.")
                    break

                max_inliers = 0
                best_inliners_mask = None

                for _ in range(iterations):
                    # 1. Sample 3 random points
                    idx = np.random.choice(points.shape[0], 3, replace=False)
                    p = points[idx]

                    # 2. Find plane au + bv + c = d
                    A = np.column_stack((p[:,0], p[:,1], np.ones(3))) # coordinates u, v
                    B = p[:,2] # depth d

                    try:
                        a, b, c = np.linalg.solve(A, B) # solve au + bv + c = d for plane parameters

                        # 3. Find set of points that lay on the plane
                        d = a * points[:,0] + b * points[:,1] + c
                        d_pred = np.abs(d - points[:,2]) # distance from plane
                        mask = d_pred <= distance_threshold
                        inliers = np.sum(mask)

                        if inliers > max_inliers:
                            max_inliers = inliers
                            best_inliners_mask = mask
                    
                    except np.linalg.LinAlgError:
                        continue

                if best_inliners_mask is not None:
                    # Take inliers and color them in output image
                    inliner_points = points[best_inliners_mask]

                    for point in inliner_points:
                        u, v = int(point[0]), int(point[1])
                        output_img[v, u] = colors[i]

                    points = points[~best_inliners_mask] # remove inliers for next iteration

            cv2.imshow('Dominant Planes', output_img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
                    

def main():
    path = r"C:\Users\Lana Kočiš\Downloads\Robotski_vid\LV3-images"
    take_pictures(path)
    print("Processing depth images...")

    find_dominant_plane(path)

if __name__ == "__main__":
    main()
    




