import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import glob

def convert_avi_to_mp4(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all AVI files in the input directory
    avi_files = glob.glob(os.path.join(input_dir, "*.avi"))
    
    if not avi_files:
        print("No AVI files found in the specified directory.")
        return
    
    # Process each AVI file
    for avi_file in avi_files:
        try:
            # Get the base filename without extension
            base_name = os.path.splitext(os.path.basename(avi_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}.mp4")
            
            print(f"Converting {base_name}.avi to {base_name}.mp4...")
            
            # Load the video file
            video = VideoFileClip(avi_file)
            
            # Convert and save as MP4
            video.write_videofile(output_file, codec="libx264", audio_codec="aac")
            
            # Close the video file
            video.close()
            
            print(f"Successfully converted {base_name}.avi to {base_name}.mp4")
            
        except Exception as e:
            print(f"Error converting {avi_file}: {str(e)}")
            continue

def main():
    # Specify input and output directories
    input_directory = "/public/home/liuhuijie/dits/dataset/preprocess_ffs/test/videos"
    output_directory = "/public/home/liuhuijie/dits/Latte/test/real_test_videos"
    
    # Validate input directory
    if not os.path.isdir(input_directory):
        print("Error: Input directory does not exist.")
        return
    
    convert_avi_to_mp4(input_directory, output_directory)
    print("Conversion process completed.")

if __name__ == "__main__":
    main()