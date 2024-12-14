from app import annotate_faces_in_video

def main():
    # Path to the video file
    weights_path = "tmp/checkpoints/ssd7_epoch-13_loss-2.0928_val_loss-2.1003.keras"
    img_height = 512
    img_width = 512
    video_name = "Obama jokingly calls Biden Vice President Biden during White House event.mp4"
    annotate_faces_in_video(weights_path, img_height, img_width, video_name)
    print("Program is running")

if __name__ == "__main__":
    main()