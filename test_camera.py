import os
import time
import numpy as np
from primesense import openni2
from primesense import _openni2
import cv2


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

def main():
    path = r"D:\Robotski_vid\LV4_images"
    for obj in range(1, 4):
        take_pictures(path, obj)

if __name__ == "__main__":
    main()