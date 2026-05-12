import os
from primesense import openni2
from primesense import _openni2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

def take_pictures(path, object_id):
    obj_path = os.path.join(path, f"object_{object_id}")
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)

    openni2.initialize(r"C:\Program Files\OpenNI2\Redist")
    dev = openni2.Device.open_any()
    
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()
    
    mode = _openni2.OniVideoMode(pixelFormat=_openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640, resolutionY=480, fps=30)
    depth_stream.set_video_mode(mode)
    color_mode = _openni2.OniVideoMode(pixelFormat=_openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30)
    color_stream.set_video_mode(color_mode)
   
    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    
    depth_stream.start()
    color_stream.start()

    count = 0
    print(f"Filming object {object_id}. Take picture: SPACE, Exit: ESC")

    try:
        while count < 5:
            d_frame = depth_stream.read_frame()
            c_frame = color_stream.read_frame()

            d_data = np.ndarray((d_frame.height, d_frame.width), dtype=np.uint16, buffer=d_frame.get_buffer_as_uint16())
            c_data = np.ndarray((c_frame.height, c_frame.width, 3), dtype=np.uint8, buffer=c_frame.get_buffer_as_uint8())
            c_data = cv2.cvtColor(c_data, cv2.COLOR_RGB2BGR)

            cv2.imshow('Camera - Live', c_data)
            key = cv2.waitKey(1)
            
            if key == ord(' '):
                cv2.imwrite(os.path.join(obj_path, f"sl-{count:05d}.bmp"), c_data)
                np.save(os.path.join(obj_path, f"sl-{count:05d}-D.txt"), d_data)
                print(f"Picture saved {count}")
                count += 1
            elif key == 27:
                break
    finally:
        depth_stream.stop()
        color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()

        
def main():
    path = r"C:\Users\Lana Kočiš\Downloads\Robotski_vid\LV4_images"
    for obj_id in range(1, 4):
        take_pictures(path, obj_id)

if __name__ == "__main__":
    main()