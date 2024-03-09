from flask import Flask, render_template, Response
import cv2 as cv
import numpy as np
import base64
import threading
import os
from datetime import timedelta
from flask import Flask, render_template, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

# def object_detection(input_video, output_video):
#     classes = ['D00', 'D10', 'D20', 'D40']
#     np.random.seed(42)
#     COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
#     weightsPath = "yolo.weights"
#     configPath = "yolov3.cfg"
#     net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
#     ln = net.getLayerNames()
#     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#     vs = cv.VideoCapture(os.path.join('static', 'sample_input.mp4'))
#     writer = None
#     (W, H) = (None, None)

#     try:
#         prop = cv.CAP_PROP_FRAME_COUNT
#         total = int(vs.get(prop))
#         print("[INFO] {} total frames in video".format(total))
#     except:
#         print("[INFO] could not determine # of frames in video")
#         print("[INFO] no approx. completion time can be provided")
#         total = -1

#     while True:
#         (grabbed, frame) = vs.read()
#         if not grabbed:
#             break

#         if W is None or H is None:
#             (H, W) = frame.shape[:2]

#         blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
#         net.setInput(blob)
#         start = time.time()
#         layerOutputs = net.forward(ln)
#         end = time.time()

#         boxes = []
#         confidences = []
#         classIDs = []

#         for output in layerOutputs:
#             for detection in output:
#                 scores = detection[5:]
#                 classID = np.argmax(scores)
#                 confidence = scores[classID]
#                 if confidence > 0.3:
#                     box = detection[0:4] * np.array([W, H, W, H])
#                     (centerX, centerY, width, height) = box.astype("int")
#                     x = int(centerX - (width / 2))
#                     y = int(centerY - (height / 2))
#                     boxes.append([x, y, int(width), int(height)])
#                     confidences.append(float(confidence))
#                     classIDs.append(classID)

#         idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.3, 0.25)

#         if writer is None:
#             # Use a relative path within the 'static' folder
#             output_path = os.path.join('static', 'result.mp4')
#             fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
#             writer = cv.VideoWriter(output_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
#             if total > 0:
#                 elap = (end - start)
#                 print("[INFO] single frame took {:.4f} seconds".format(elap))
#                 print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

#         writer.write(frame)

#     print("[INFO] cleaning up...")
#     writer.release()
#     vs.release()

# Route to start object detection
@app.route('/start_detection')
def start_detection():
    input_video = 'static/sample_input.mp4'  # Change this to your input video path
    output_video = 'static/result.avi'  # Change this to your output video path
    object_detection(input_video, output_video)
    return 'Object detection started!'

# Route for video streaming
def gen_frames():
    output_video = 'static/result.avi'  # Change this to your output video path
    cap = cv2.VideoCapture(output_video)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

