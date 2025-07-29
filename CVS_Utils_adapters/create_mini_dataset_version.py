#This file creates a subset of the complete CVS sages dataset, just for faster debbuging
# In this case, only need new json files with less key frames, just 2 for each video

import os
import glob
import json
import pandas as pd
import random


random.seed(42)

def subset_creation(path: str):
    """
    This fucntion expects the path to the json file, load all info in a dataframe and randomly selects 2 frames from each video.
    Then saves this info in a new json file with the "mini" prefix
    """

    with open(path, 'r') as f:
        data = json.load(f)

    df_images = pd.DataFrame(data['images'])
    df_annots = pd.DataFrame(data['annotations'])

    selected_images = pd.DataFrame(columns=df_images.columns)
    selected_annots = pd.DataFrame(columns=df_annots.columns)


    for video_idx in df_images['video_name'].unique().tolist():

        possible_frames = df_images[df_images['video_name'] == video_idx]['file_name'].tolist()
        nums = random.sample(range(18), 2)
        selected_frame_1 = possible_frames[nums[0]]
        selected_frame_2 = possible_frames[nums[1]]

        for image_name in [selected_frame_1, selected_frame_2]:
            image_info = df_images[df_images['file_name'] == image_name]
            selected_images = pd.concat([selected_images, image_info], ignore_index=True)

            annot_info = df_annots[df_annots['image_name'] == image_name]
            selected_annots = pd.concat([selected_annots, annot_info], ignore_index=True)

    selected_images = selected_images.sort_values(by='file_name')
    selected_annots = selected_annots.sort_values(by='image_name')
    list_of_images = selected_images.to_dict(orient='records')
    list_of_annots = selected_annots.to_dict(orient='records')

    complete_data = {'images': list_of_images, 'annotations': list_of_annots}
    file_name = os.path.basename(path)
    dir_name = os.path.dirname(path)

    with open(os.path.join(dir_name, f'mini_{file_name}'), 'w') as f:
        json.dump(complete_data, f, indent=4)

    print(f'Mini version of {file_name} created succesfully')



if __name__ == '__main__':
    
    json_files_paths = glob.glob(os.path.join('/home/dlmanrique/Endovis/CVS_Challenge/Models/MuST-Challenge-CVS-2025/data/Cvssages/only_challenge_annot_data/annotations', '*.json'))
    
    for pth in sorted(json_files_paths):
        subset_creation(pth)