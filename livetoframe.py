import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import time



def make_darker(frame, alpha=0.7, beta=20):
    # Apply contrast and brightness adjustment to the entire frame
    darker_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return darker_frame



def rewriteFrames(input_video_path,output_video_path):

    # Input video file path
    input_video_path = "static/sample_input.mp4"

    # Output video file path
    output_video_path = "static/output_video_30fps.mp4"

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Get the original video's frame rate and total frames
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames to take (1 frame per second)
    frames_to_take = int(original_fps)  # Change this if you want a different rate

    # Create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (int(cap.get(3)), int(cap.get(4))))

    # Loop through the frames of the original video and save frames
    for frame_number in range(0, total_frames, int(original_fps)):
        # Set the video capture's position to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the current frame
        ret, frame = cap.read()

        # Check if the frame is successfully read
        if not ret:
            print(f"Error reading frame {frame_number}.")
            break

        # Write the frame to the output video
        out.write(frame)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()




def rewriteFrameswithoutSave():
    # Input video file path
    input_video_path = "static/sample_input.mp4"

   # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Get the original video's frame rate and total frames
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames to take (1 frame per second)
    frames_to_take = int(original_fps)  # Change this if you want a different rate

    # List to store frames
    frames_list = []

    # Loop through the frames of the original video and store frames to the list
    for frame_number in range(0, total_frames, int(original_fps)):
        # Set the video capture's position to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the current frame
        ret, frame = cap.read()

        # Check if the frame is successfully read
        if not ret:
            print(f"Error reading frame {frame_number}.")
            break

        # Store the frame in the list
        frames_list.append(frame)

    # Display the frames as a video
    for frame in frames_list:
        cv2.imshow("Video Collage", frame)
        
        # Adjust the delay to control the frame rate
        delay = int(1000 / original_fps)
        
        # Wait for a key press or delay to simulate video playback
        if cv2.waitKey(delay) & 0xFF == 27:  # Press 'Esc' to exit the loop
            break

    # Release the VideoCapture object
    cap.release()

def imagesinCollage():
    # Input video file path
    input_video_path = "static/sample_input.mp4"

   
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Get the original video's frame rate and total frames
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the number of frames to take (1 frame per second)
    frames_to_take = 30  # Adjust this number as needed

    # Calculate the step size to evenly distribute frames throughout the video
    step_size = total_frames // frames_to_take

    # List to store frames
    frames_list = []

    # Loop through the frames of the original video and store frames to the list
    for frame_number in range(0, total_frames, step_size):
        # Set the video capture's position to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the current frame
        ret, frame = cap.read()

        frame = func(frame)

        # Check if the frame is successfully read
        if not ret:
            print(f"Error reading frame {frame_number}.")
            break

        # Store the frame in the list
        frames_list.append(frame)

    # Concatenate frames horizontally to create a collage
    collage = np.concatenate(frames_list, axis=1)

    # Display the collage
    cv2.imshow("Video Collage", collage)

    # Wait for a key press to close the window
    cv2.waitKey(0)

    # Release the VideoCapture object
    cap.release()

def func(frame):
    classes = ['D00', 'D10', 'D20', 'D40']
    alt_names = {'D00': 'lateral_crack', 'D10': 'linear_cracks', 'D20': 'aligator_crakcs', 'D40': 'potholes'}
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    # derive the paths to the YOLO weights and model configuration
    weightsPath = "yolo.weights"
    configPath = "yolov3.cfg"
    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # read the next frame from the file
    # frame = cv2.imread(input_file)
    (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
  
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.3:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.25)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = classes[classIDs[i]]
            text = "{}: {:.4f}".format(alt_names[label], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image using matplotlib
    cv2.imshow("YOLO Object Detection",cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # cv2.show()


def detectLive():
        
    # Set the video source (0 for default camera)
    video_source = "static/sample_input.mp4"

    # rewriteFrames(video_source,"static/output_video_30fps.mp4",5)

    # Set the frames per second (fps)
    fps = 30

    # Create a VideoCapture object
    cap = cv2.VideoCapture("static/sample_input.mp4")

    # Check if the video capture is successful
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()


    # Set the starting time in seconds (e.g., 10 seconds)
    start_time_seconds = 60

    # Set the starting frame index based on the desired start time
    start_frame_index = int(start_time_seconds * fps)

    # Set the current frame index to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

    # Create a directory to save the frames
    output_dir = "frames"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize frame counter
    frame_count = 0
    local_fps = start_frame_index
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Check if the frame is successfully captured
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the live capture (optional)
        # cv2.imshow("Live Capture", frame)

        # Make the entire image darker
        
        # alpha = input("alpha: ")
        # beta = input("alpha: ")
        # darker_frame = make_darker(frame.copy(), alpha=1 , beta=1  )
        

        # Display the adjusted frame
        func(frame)
        local_fps +=  4
        cap.set(cv2.CAP_PROP_POS_FRAMES,  local_fps)

        # Save the frame as a JPG image
        # frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        # cv2.imwrite(frame_filename, frame)

        # Increment frame counter
        frame_count += 1

        # Wait for the specified time to achieve the desired fps
        delay = int(1000 / fps)
        if cv2.waitKey(delay) & 0xFF == 27:  # Press 'Esc' to exit the loop
            break

        # Release the VideoCapture and close any open windows
    cap.release()
    cv2.destroyAllWindows()

detectLive()
# imagesinCollage()