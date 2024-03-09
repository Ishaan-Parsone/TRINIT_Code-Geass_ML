import cv2
import numpy as np


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

rewriteFrameswithoutSave()

# Close any open windows
cv2.destroyAllWindows()
