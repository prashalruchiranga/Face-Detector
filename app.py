import cv2
from face_detector import FaceDetector
import numpy as np
import tensorflow as tf
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
import matplotlib.pyplot as plt

def annotate_faces_in_video(weights_path, img_height, img_width, input_video_path, output_video_path):
    n_classes = 1
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4

    # Set up the VideoWriter to save the output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize a FaceDetector object
    detector = FaceDetector(img_height, img_width, weights_path)

    # Loop to play the video frame by frame
    while cap.isOpened():
        ret, frame_bgr = cap.read()  # Read a frame
        if not ret:
            print("Reached the end of the video or encountered an error.")
            break

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        input_images = []
        frame_resized = tf.image.resize(frame, (img_height, img_width))
        frame_resized = frame_resized.numpy()
        input_images.append(frame_resized)
        input_images = np.array(input_images)

        y_pred = detector.predict_bbxs(input_images)

        y_pred_decoded = decode_detections(y_pred,
                                        confidence_thresh=0.5,
                                        iou_threshold=0.2,
                                        top_k=200,
                                        normalize_coords=True,
                                        img_height=img_height,
                                        img_width=img_width
                                        )
        
        # Display the image and draw the predicted boxes onto it.

        # Set the colors for the bounding boxes
        colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
        classes = ['background', 'face']

        for box in y_pred_decoded[0]:
            # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
            xmin = box[2] * frame.shape[1] / img_width
            ymin = box[3] * frame.shape[0] / img_height
            xmax = box[4] * frame.shape[1] / img_width
            ymax = box[5] * frame.shape[0] / img_height
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])

            # Draw bounding box on the frame
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255), thickness=2)

            # Get text size to calculate padding
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Calculate background rectangle coordinates
            label_bg_xmin = xmin
            label_bg_ymin = ymin - text_height - baseline - 4  # Add padding above the box
            label_bg_xmax = xmin + text_width + 4
            label_bg_ymax = ymin

            # Draw the label background
            cv2.rectangle(frame, (int(label_bg_xmin), int(label_bg_ymin)), (int(label_bg_xmax), int(label_bg_ymax)), (0, 0, 255), thickness=cv2.FILLED)

            # Draw the text
            cv2.putText(frame, label, (int(xmin), int(ymin) - 5), font, font_scale, (255, 255, 255), thickness)

            #cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert frame back to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write the transformed frame to the output video
        out.write(frame_bgr)

        # Display the frame with bounding boxes
        cv2.imshow('Video Player', frame_bgr)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    print(f'Output video saved to {output_video_path}')
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
