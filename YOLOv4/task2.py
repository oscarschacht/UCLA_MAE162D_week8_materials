# import useful libraries
import cv2
import subprocess
import os
from yolo_utils import *
from picamera2 import Picamera2

# video file names
temp_video = "temp_recording.avi"
output_video = "recording.mp4"

# check OpenCV + CUDA
print("OpenCV version :", cv2.__version__)
print("Available CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount(), "\n")

# load class names
obj_file = './obj.names'
classNames = read_classes(obj_file)
print("Classes' names :", classNames, "\n")

# load YOLO model
modelConfig_path = './cfg/yolov4.cfg'
modelWeights_path = './weights/yolov4.weights'

neural_net = cv2.dnn.readNetFromDarknet(modelConfig_path, modelWeights_path)
neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

confidenceThreshold = 0.5
nmsThreshold = 0.1

network = neural_net
height, width = 128, 128   # input size for network

# initialize Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={'size': (640, 480)}))
picam2.start()

# setup Video Writer (AVI first)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(temp_video, fourcc, 30.0, (640, 480))

print("[MAIN] Recording started... Press Ctrl+C to stop.")

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # object detection
        outputs = convert_to_blob(frame, network, height, width)
        bounding_boxes, class_objects, confidence_probs = object_detection(
            outputs, frame, confidenceThreshold)

        for i in range(len(bounding_boxes)):
            print(f"[Debug] Detected: Class={class_objects[i]}, Confidence={confidence_probs[i]:.2f}")
            # TODO: change the class number to the class number of traffic light in obj.names file
            if class_objects[i] == 3:
                # Step 1: crop bounding box
                x, y, w, h = bounding_boxes[i]
                cropped = frame[y:y+h, x:x+w]

                if cropped.size == 0:
                    continue
        
                # Step 2: convert to HSV
                hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        
                # Step 3: create red mask (single range 0–10)
                lower_red = np.array([0, 120, 40])
                upper_red = np.array([10, 255, 255])
                red_mask = cv2.inRange(hsv, lower_red, upper_red)
        
                # Step 4: check contour area
                contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
                for cnt in contours:
                    if cv2.contourArea(cnt) > 50:
                        print("Red light detected!")
                        break
                pass
        

        indices = nms_bbox(
            bounding_boxes,
            confidence_probs,
            confidenceThreshold,
            nmsThreshold
        )

        box_drawing(
            frame,
            indices,
            bounding_boxes,
            class_objects,
            confidence_probs,
            classNames,
            color=(0, 255, 255),
            thickness=2
        )

        # write frame to video file
        out.write(frame)

except KeyboardInterrupt:
    print("\n[MAIN] Stopping recording...")

# cleanup
out.release()
picam2.close()

print("[MAIN] Converting to MP4 using ffmpeg...")

subprocess.run(["ffmpeg", "-y", "-i", temp_video, "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.remove(temp_video)

print("[MAIN] Video saved successfully as", output_video)
