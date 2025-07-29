#This file creates a subset of the complete CVS sages dataset, just for faster debbuging
# In this case, only need new json files with less key frames, just 2 for each video

import os
import glob
import json


def subset_creation(path: str):
    """
    This fucntion expects the path to the json file, load all info in a dataframe and randomly selects 2 frames from each video.
    Then saves this info in a new json file with the "mini" prefix
    """

    with open(path, 'r') as f:
        data = json.load(f)
    breakpoint()


if __name__ == '__main__':
    
    json_files_paths = glob.glob(os.path.join('/home/dlmanrique/Endovis/CVS_Challenge/Models/MuST-Challenge-CVS-2025/data/Cvssages/only_challenge_annot_data/annotations', '*.json'))
    
    for pth in sorted(json_files_paths):
        subset_creation(pth)