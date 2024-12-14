from app import annotate_faces_in_video

def main():
    print("Program is running")
    # Path to the video file
    weights_path = "tmp/checkpoints/ssd7_epoch-13_loss-2.0928_val_loss-2.1003.keras"
    img_height = 512
    img_width = 512
    input_video_path = "examples/inputs/Obama_jokingly_calls_Biden_Vice_President_Biden_during_White_House_event.mp4"
    output_video_path = "examples/outputs/Obama_jokingly_calls_Biden_Vice_President_Biden_during_White_House_event_annotated.mp4"
    annotate_faces_in_video(weights_path, img_height, img_width, input_video_path, output_video_path)

if __name__ == "__main__":
    main()