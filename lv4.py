import os
from primesense import openni2
from primesense import _openni2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
#import teaserpp_python
import time

def take_pictures(path, obj):
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
                cv2.imwrite(os.path.join(path, f"obj-{obj:05d}-sl-{count:05d}.bmp"), color_image)
                np.savetxt(os.path.join(path, f"obj-{obj:05d}-sl-{count:05d}-D.txt"), depth_image, fmt='%d')
                print(f"Saved {count}/10")
                count += 1
            elif key == 27:
                break
    finally:
        depth_stream.stop()
        color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()


def create_and_preprocess_pcds(object):
    obj_path = os.path.join(r"D:\Robotski_vid\LV4_images", f"object_{object}")
    pcds = []

    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    for i in range(10):
        color_path = os.path.join(obj_path, f"obj-{object:05d}-sl-{i:05d}.png")
        depth_path = os.path.join(obj_path, f"obj-{object:05d}-sl-{i:05d}-D.txt")

        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            print(f"File {color_path} or {depth_path} does not exist!")
            continue

        color_img = o3d.io.read_image(color_path)
        depth_data = np.loadtxt(depth_path, dtype=np.uint16)
        depth_img = o3d.geometry.Image(depth_data)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, depth_scale=1000.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # 1. Poduzorkovanje 
        pcd = pcd.voxel_down_sample(voxel_size=0.005) 
        
        # 2. Čišćenje šuma 
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        pcds.append(pcd)
        print(f"Processed point cloud {i}")

    return pcds

def convert_bmp_to_png(folder):
    for f in os.listdir(folder):
        if f.endswith(".bmp"):
            path = os.path.join(folder, f)
            img = cv2.imread(path)
            cv2.imwrite(path.replace(".bmp", ".png"), img)
    print("Conversion done.")

def FPFH_registration(source, target):
    # a) Detection of descriptors using FPFH
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)) # takes points in radius 0.1 and max 30 neighbors to estimate normals
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    desc_source = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))
    desc_target = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))

    # b)Registration using TEASER++ algorithm
    src_pts = np.asarray(desc_source.points).T # turning FPFH descriptors into array 
    tgt_pts = np.asarray(desc_target.points).T

    n_neigh = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tgt_pts) # finding nearest neighbor in target for each source point
    distances, indices = n_neigh.kneighbors(src_pts)

    src_pts = np.asarray(source.points)[np.arange(len(indices))].T # getting real 3D points from source point cloud corresponding to the FPFH descriptors
    tgt_pts = np.asarray(target.points)[indices.flatten()].T

    #params = teaserpp_python.RobustRegistrationSolverParams(
    #    cbar2=1,
    #    noise_bound=1,
    #    estimate_scaling=True,
    #    rotation_estimation_algorithm=teaserpp_python.RotationEstimationAlgorithm.GNC_TLS,
    #    rotation_gnc_factor=1.4,
    #    rotation_max_iterations=100,
    #    rotation_cost_threshold=1e-12
    #)
    #params.noise_bound = 0.05 # threshold for noise tolerance
    #params.estimate_scaling = False # RGB-D images do not have scaling issues, so we set this to False
    #params.rotation_estimation_algorithm = teaserpp_python.RotationEstimationAlgorithm.GNC_TLS # method for ignoring outliers in rotation estimation

    #solver = teaserpp_python.RobustRegistrationSolver(params) # initializing TEASER++ solver with the defined parameters
    #solver.solve(src_pts, tgt_pts) # takes paired points and finds the best transformation that aligns them
    #solution = solver.getSolution() # getting the resulting transformation parameters (rotation matrix and translation vector) - homogeneous transformation matrix

    # c) Calculate transformation matrix
    transformation = np.eye(4)
    #transformation[:3, :3] = solution.rotation
    #transformation[:3, 3] = solution.translation

    # d) Transform the source and refresh the target point cloud
    source.transform(transformation)
    updated_target = target + source
    o3d.visualization.draw_geometries([updated_target], window_name="FPFH Registration Result")

    return updated_target, transformation

def RANSAC_and_ICP(source, target):
    # a) Global registration of source and target using RANSAC to find dominant planes
    distance = 0.002
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=30))

    desc_source = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))
    desc_target = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))
    
    # global registration which moves point clouds close to each other, but not perfectly aligned, so that ICP can do the fine-tuning in the next step
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, desc_source, desc_target, True, distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, # number of points to sample for plane estimation
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), # checks if triangles formed by sampled points have similar edge lengths in source and target
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance)], # checks if corresponding points are within a certain distance
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999) # maximum number of iterations and confidence level for RANSAC convergence
    )

    # b) Local registration using ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane() # ICP fine-tuning using point-to-plane distance, which is more accurate for RGB-D data
    )


    # c), d) Transform the source and refresh the target point cloud
    source.transform(result_icp.transformation)
    updated_target = target + source

    return updated_target, result_icp.transformation  

        
def main():
    path = r"D:\Robotski_vid\LV4_images"

    convert_bmp_to_png(r"D:\Robotski_vid\LV4_images\object_1")
    convert_bmp_to_png(r"D:\Robotski_vid\LV4_images\object_2")
    convert_bmp_to_png(r"D:\Robotski_vid\LV4_images\object_3")

    for obj in range(1, 4):
        #take_pictures(path, obj)
       
        cloud_list = create_and_preprocess_pcds(obj)
        if not cloud_list: 
            print(f"No point clouds created for object {obj}. Skipping to next object.")
            continue

        print(f"Object {obj}: Target set. Ready for FPFH and TEASER++.")

        # FPFH + TEASER++
        #print(f"Object {obj}: Starting FPFH + TEASER++ registration.")
        #target_teaser = cloud_list[0]
        #start_teaser = time.time()

        #for i in range(1, len(cloud_list)):
        #    target_teaser, _ = FPFH_registration(cloud_list[i], target_teaser)

        #end_teaser = time.time()
        #print(f"FPFH + TEASER++ time: {end_teaser - start_teaser:.2f} seconds.")

        # RANSAC + ICP
        print(f"Object {obj}: Starting RANSAC + ICP registration.")
        target_ransac = cloud_list[0]
        start_ransac = time.time()

        for i in range(1, len(cloud_list)):
            target_ransac, _ = RANSAC_and_ICP(cloud_list[i], target_ransac)
            
        end_ransac = time.time()
        print(f"RANSAC + ICP time: {end_ransac - start_ransac:.2f} seconds.")

        #o3d.visualization.draw_geometries([target_teaser], window_name=f"Final FPFH + TEASER++ Result for Object {obj}")
        o3d.visualization.draw_geometries([target_ransac], window_name=f"Final RANSAC + ICP Result for Object {obj}")

if __name__ == "__main__":
    main()