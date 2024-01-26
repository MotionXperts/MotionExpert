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

def format_json(json_file,filename,index):
  """
  Input: a json filename
  Output: a dictionary in the format of <(start_time, end_time), (id, context)>
  """

  with open(json_file, 'r') as file:
    data = json.load(file)
    clip = data[index]
    strs = clip['timestamp'].split('-')
    start_time = float(strs[0].strip())
    end_time = float(strs[1].strip())
    # if (start_time, end_time) in output:
    #  output[(start_time, end_time)] = (output[(start_time, end_time)][0], output[(start_time, end_time)][1] + ' ' + clip['context'].strip())
    # else:
    print(clip)
    output_list = [ { "id": clip['id'], "timestamp": f"{start_time} - {end_time}", "context": clip['context'] }]
    with open(filename, 'w') as json_file:
      json.dump(output_list, json_file, indent=4)


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
    if logging:
      print(f"[Configuration] Video: {input_video_file}, Json: {json_file}")
    mp4_base, mp4_ext = os.path.splitext(mp4_file)
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    with open(json_file, 'r') as file:
      data = json.load(file)
      index = 0
      for clip in data:
        output_video_file = os.path.join(output_folder, (mp4_base + '_' + str(clip['id']) + mp4_ext))
        json_output_file = os.path.join(output_folder, (mp4_base + '_' + str(clip['id']) + '.json'))
        strs = clip['timestamp'].split('-')
        start_time = float(strs[0].strip())
        end_time = float(strs[1].strip())
        if logging:
          print(f"Trimming {clip['id']} video {input_video_file} to {output_video_file}")
        ffmpeg_extract_subclip(input_video_file, start_time, end_time, targetname=output_video_file)

        format_json(json_file,json_output_file,index)
        index+=1
        # write_json(formatted_json, json_output_file)  # Write the json file to the destinated folder

folders = [f for f in os.listdir(input_folder_path) if os.path.isdir(os.path.join(input_folder_path, f))]
for folder in folders:
  folder_path = os.path.join(input_folder_path, folder)
  files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
  trim_video(folder_path, files, os.path.join(output_folder_path, folder))


# Command: python /utils/trim_video_unit.py --input_folder="/home/weihsin/datasets/Axel2" --output_folder="/home/weihsin/datasets/Axel2/test" --verbose