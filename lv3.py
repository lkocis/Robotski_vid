import numpy as np
import cv2
from read_kinect_pic import read_kinect_pic
import os
from primesense import openni2
from primesense import _openni2
import time

def take_pictures(path):
    if not os.path.exists(path):
        os.makedirs(path)

    openni2.initialize(r"C:\Program Files\OpenNI2\Redist") 
    dev = openni2.Device.open_any()
    
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()
    
    depth_stream.set_video_mode(_openni2.OniVideoMode(pixelFormat=_openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=320, resolutionY=240, fps=30))
    color_stream.set_video_mode(_openni2.OniVideoMode(pixelFormat=_openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))
    
    depth_stream.start()
    color_stream.start()

    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    count = 0
    print("ASUS Xtion started. Press SPACE to take a picture, ESC to exit.")
    
    time.sleep(2) 

    try:
        while count < 10:
            try:
                openni2.wait_for_any_stream([depth_stream, color_stream])
            except _openni2.OpenNIError:
                continue 

            d_frame = depth_stream.read_frame()
            c_frame = color_stream.read_frame()

            d_data = d_frame.get_buffer_as_uint16()
            depth_image = np.ndarray((d_frame.height, d_frame.width), dtype=np.uint16, buffer=d_data)

            c_data = c_frame.get_buffer_as_uint8()
            color_image = np.ndarray((c_frame.height, c_frame.width, 3), dtype=np.uint8, buffer=c_data)
            
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            cv2.imshow('Color Image', color_image)
            
            # WaitKey gives more time for the stream to stabilize and ensures the window is responsive
            key = cv2.waitKey(100) & 0xFF 
            
            if key == ord(' '):
                cv2.imwrite(os.path.join(path, f"sl-{count:05d}.bmp"), color_image)
                np.savetxt(os.path.join(path, f"sl-{count:05d}-D.txt"), depth_image, fmt='%d')
                print(f"Saved {count}/10")
                count += 1
            elif key == 27:
                break
    finally:
        depth_stream.stop()
        color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()

def find_dominant_plane(folder_path):
    img_shape = (240, 320)

    # RANSAC parameters
    iterations = 1000
    num_planes = 3
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # red, green, blue

    for filename in os.listdir(folder_path):
        if filename.endswith("-D.txt"):
            depth_path = os.path.join(folder_path, filename)
            depth_image, point_3d_array, n_3d_points = read_kinect_pic(depth_path, img_shape)

            color_filename = filename.replace("-D.txt", ".bmp")
            color_path = os.path.join(folder_path, color_filename)
            if os.path.exists(color_path):
                color_image = cv2.imread(color_path)
                cv2.imshow('Color Image', color_image)
                cv2.imwrite(os.path.join(r"D:\Robotski_vid\LV3_planes", f"{filename[:-6]}-color.png"), color_image)
            cv2.imshow('Depth Image', depth_image)
            cv2.imwrite(os.path.join(r"D:\Robotski_vid\LV3_planes", f"{filename[:-6]}-depth.png"), depth_image)

            print(f"Processed {filename}:")

            points = np.array(point_3d_array)
            points = points[points[:, 2] > 0] # filtering missing values

            # Calculate depth range (max distance of point from plane to be considered an inlier)
            depth_range = points[:,2].max() - points[:,2].min()
            distance_threshold = depth_range * 0.03 # 3% of depth range

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

                        # 4., 5. If this plane has more inliers than the best one so far, update best plane
                        if inliers > max_inliers:
                            max_inliers = inliers
                            best_inliners_mask = mask
                    
                    except np.linalg.LinAlgError:
                        continue
                
                # 6. If we found a plane with inliers, color those inliers in the output image
                if best_inliners_mask is not None:
                    inliner_points = points[best_inliners_mask]


                    for point in inliner_points:
                        u, v = int(point[0]), int(point[1])
                        output_img[v, u] = colors[i]

                    points = points[~best_inliners_mask] # remove inliers for next iteration

            cv2.imwrite(os.path.join(r"D:\Robotski_vid\LV3_planes", f"{filename[:-6]}-planes.png"), output_img)
            cv2.imshow('Dominant Planes', output_img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
                    

def main():
    path = r"D:\Robotski_vid\LV3_images"
    #take_pictures(path)
    print("Processing depth images...")

    find_dominant_plane(path)

if __name__ == "__main__":
    main()
    




