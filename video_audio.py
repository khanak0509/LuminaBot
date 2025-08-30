from moviepy.editor import VideoFileClip

video = VideoFileClip("What is AI Engineering.mp4")
audio = video.audio
audio.write_audiofile("What is AI Engineering.mp3")
