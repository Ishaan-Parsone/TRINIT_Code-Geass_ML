from flask import Flask, render_template, Response
import cv2 as cv
import numpy as np
import base64
import threading
import os
from flask import session, g
from datetime import timedelta

app = Flask(__name__)
app.secret_key = '6e32fb8fceefd0b085f4246855f33a87'


# Function for object detection
def object_detection(input_video, output_video):
    import time

    classes = ['D00', 'D10', 'D20', 'D40']
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    weightsPath = "yolo.weights"
    configPath = "yolov3.cfg"
    net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    vs = cv.VideoCapture(os.path.join('static', 'sample_input.mp4'))
    writer = None
    (W, H) = (None, None)

    try:
        prop = cv.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.3:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.3, 0.25)

        if writer is None:
            # Use a relative path within the 'static' folder
            output_path = os.path.join('static', 'result.mp4')
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter(output_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

        writer.write(frame)

    print("[INFO] cleaning up...")
    writer.release()
    vs.release()

# Function to read frames and convert them to base64
def generate_frames():
    vs = cv.VideoCapture(os.path.join('static', 'result.mp4'))

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break

        # Convert the frame to base64
        _, buffer = cv.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_base64 + b'\r\n\r\n')


    vs.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')


# Main
if __name__ == '__main__':
    # Start the object detection in a separate thread
    detection_thread = threading.Thread(target=object_detection, args=('static/sample_input.mp4', 'result.mp4'))
    detection_thread.start()

    # Run the Flask app
    app.run(debug=True)
