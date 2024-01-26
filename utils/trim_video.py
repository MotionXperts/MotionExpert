"""
Intent: Trim the videos inside the given folder into multiple clips based on the json file provided
Author: Tom
Date: Sept. 13, 2023
"""

import argparse
import json
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Folder path parser')

# Add the --input_folder and --output_folder arguments
parser.add_argument('--input_folder', type=str, help='Input folder of the videos')
parser.add_argument('--output_folder', type=str, help='Output folder for the trimmed videos')
parser.add_argument('--verbose', action='store_true', help='Check if the tag "log" is given')

# Parse the command-line arguments
args = parser.parse_args()

input_folder_path = None
output_folder_path = None
logging = False

if args.input_folder:
  input_folder_path = args.input_folder
if args.output_folder:
  output_folder_path = args.output_folder
if args.verbose:
  logging = True

def trim_video(input_folder, files, output_folder):
  json_file = None
  mp4_file = None
  for file in files:
    if file.endswith('.json'):
      json_file = os.path.join(input_folder, file)
    if file.endswith('.mp4'):
      mp4_file = file

  if mp4_file and json_file:
    input_video_file = os.path.join(input_folder, mp4_file)
    if logging:
      print(f"[Configuration] Video: {input_video_file}, Json: {json_file}")
    mp4_base, mp4_ext = os.path.splitext(mp4_file)
    json_output_file = os.path.join(output_folder, (mp4_base + '.json'))
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)

    with open(json_file, 'r') as file:
      data = json.load(file)
      with open(json_output_file, 'w') as json_file:  # Write the same json file with a different name
        json.dump(data, json_file, indent=4)
      for clip in data:
        output_video_file = os.path.join(output_folder, (mp4_base + '_' + str(clip['id']) + mp4_ext))
        strs = clip['timestamp'].split('-')
        start_time = float(strs[0].strip())
        end_time = float(strs[1].strip())
        if logging:
          print(f"Trimming {clip['id']} video {input_video_file} to {output_video_file}")
        ffmpeg_extract_subclip(input_video_file, start_time, end_time, targetname=output_video_file)

folders = [f for f in os.listdir(input_folder_path) if os.path.isdir(os.path.join(input_folder_path, f))]
for folder in folders:
  folder_path = os.path.join(input_folder_path, folder)
  files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
  trim_video(folder_path, files, os.path.join(output_folder_path, folder))

# Command: python MotionExpert/utils/trim_video.py --input_folder="datasets/Axel2" --output_folder="test" --verbose