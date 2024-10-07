from moviepy.editor import VideoFileClip

def convert_mov_to_mp4(input_file, output_file):
    # Load the video file
    video = VideoFileClip(input_file)
    
    # Write the video file in mp4 format
    video.write_videofile(output_file, codec='libx264', audio_codec='aac')

# Example usage
# input_mov_file = 'standard/all_correct.MOV'
# output_mp4_file = 'all_correct.mp4'
# convert_mov_to_mp4(input_mov_file, output_mp4_file)

# input_mov_file = 'standard/shaking_error.MOV'
# output_mp4_file = 'shaking_error.mp4'
# convert_mov_to_mp4(input_mov_file, output_mp4_file)

input_mov_file = 'standard/shaking_error_2.MOV'
output_mp4_file = 'shaking_error_2.mp4'
convert_mov_to_mp4(input_mov_file, output_mp4_file)

input_mov_file = 'standard/last_drop_error.MOV'
output_mp4_file = 'last_drop_error.mp4'
convert_mov_to_mp4(input_mov_file, output_mp4_file)