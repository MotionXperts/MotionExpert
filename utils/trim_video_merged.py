"""
Intent: Trim the videos inside the given folder into multiple clips based on the json file provided (with clips of the same start/end time merged)
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
parser.add_argument('--store', action='store_true', help='Check if the tag "log" is given')
parser.add_argument('--concat', '-c', action='store_true', help='Whether to concat the labels')
parser.add_argument('--verbose', '-v', action='store_true', help='Check if the tag "log" is given')

# Parse the command-line arguments
args = parser.parse_args()

input_folder_path = None
output_folder_path = None
concat = False
logging = False

if args.input_folder:
  input_folder_path = args.input_folder
if args.output_folder:
  output_folder_path = args.output_folder
if args.concat:
  concat = True
if args.verbose:
  logging = True


def format_json(json_file):
  """
  Input: a json filename
  Output: a dictionary in the format of <(start_time, end_time), (id, context)>
  """
  output = {}
  with open(json_file, 'r') as file:
    data = json.load(file)
    for clip in data:
      strs = clip['timestamp'].split('-')
      start_time = float(strs[0].strip())
      end_time = float(strs[1].strip())
      if concat: 
        if (start_time, end_time) in output:
          output[(start_time, end_time)] = (output[(start_time, end_time)][0], output[(start_time, end_time)][1] + ' ' + clip['context'].strip())
        else:
          output[(start_time, end_time)] = (clip['id'], clip['context'].strip())
      else:
        if (start_time, end_time) in output:
          output[(start_time, end_time)][1].append(clip['context'].strip())
        else:
          output[(start_time, end_time)] = (clip['id'], [clip['context'].strip()])
  return output

def write_json(formatted_json, filename):
  """
  Accepted input format: a dictionary in the format of <(start_time, end_time), (id, context)>
  Output format: {'id', 'timestamp', 'context'}
  """
  output_list = [ { "id": value[0], "timestamp": f"{key[0]} - {key[1]}", "context": value[1] } for key, value in formatted_json.items() ]
  with open(filename, 'w') as json_file:
    json.dump(output_list, json_file, indent=4)

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
    print(f"[Configuration] Video: {input_video_file}, Json: {json_file}") if logging else None
    mp4_base, mp4_ext = os.path.splitext(mp4_file)
    json_output_file = os.path.join(output_folder, (mp4_base + '.json'))
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    
    formatted_json = format_json(json_file)
    write_json(formatted_json, json_output_file)  # Write the json file to the destinated folder
    for time, value in formatted_json.items():
      output_video_file = os.path.join(output_folder, (mp4_base + '_' + str(value[0]) + mp4_ext))
      print(f"Trimming {value[0]} video {input_video_file} to {output_video_file}") if logging else None
      ffmpeg_extract_subclip(input_video_file, time[0], time[1], targetname=output_video_file)

folders = [f for f in os.listdir(input_folder_path) if os.path.isdir(os.path.join(input_folder_path, f))]
for folder in folders:
  folder_path = os.path.join(input_folder_path, folder)
  files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
  trim_video(folder_path, files, os.path.join(output_folder_path, folder))

# Command: python MotionExpert/utils/trim_video.py --input_folder="datasets/Axel2" --output_folder="test" --verbose