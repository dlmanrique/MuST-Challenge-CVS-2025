# This file creates annotations (long_term) json file and frame list file for CVS Sages data

import os
import json
import glob
import pandas as pd
from tqdm import tqdm
from PIL import Image

def load_json(file_path):
    """
    Carga un archivo JSON y devuelve su contenido como un diccionario (u otro tipo según el JSON).
    
    Args:
        file_path (str): Ruta al archivo JSON.
    
    Returns:
        dict | list | Any: Contenido del archivo JSON.
    
    Raises:
        FileNotFoundError: Si el archivo no existe.
        json.JSONDecodeError: Si el archivo no contiene JSON válido.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, file_path, indent=4):
    """
    Guarda un objeto en un archivo JSON.

    Args:
        data (Any): Diccionario, lista u otro objeto serializable en JSON.
        file_path (str): Ruta donde se guardará el archivo JSON.
        indent (int, opcional): Nivel de indentación para formato legible. Por defecto es 4.

    Raises:
        TypeError: Si el objeto no es serializable en JSON.
        IOError: Si ocurre un error al escribir el archivo.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def majority_vote_consensus(df: pd.DataFrame) -> pd.DataFrame:
    # Detectar criterios únicos: c1, c2, c3...
    criterio_prefixes = sorted(set(col.split('_')[0] for col in df.columns if '_rater' in col))
    
    for crit in criterio_prefixes:
        rater_cols = [col for col in df.columns if col.startswith(crit + '_rater')]
        
        # Calcular la moda fila por fila (axis=1)
        df[crit + '_consensus'] = df[rater_cols].mode(axis=1)[0]  # [0] toma la moda en caso de empate (valor más bajo)
    
    return df


def frame_list_creator(df: pd.DataFrame, split: str, fold: int, fps=0.2):
    """
    This function creates the frame_list.csv file extracting all info from df -> unified_frame_labels.csv
    """
    if fps == 0.2:
        df['Video_id'] = df['Video_name'].str.split('_').str[-1].astype(int)
        df['frame_path'] = df['Video_name'] + '/' + df['frame_id'].astype(str).str.zfill(5) + '.jpg'

        #Remove unncesary columns
        df = df[['Video_name', 'Video_id', 'frame_id', 'frame_path']]
        df.to_csv(f'/home/dlmanrique/Endovis/CVS_Challenge/Models/MuST-Challenge-CVS-2025/data/Cvssages/only_challenge_annot_data/frame_lists/fold_{fold}_{split}_fps_02.csv', 
                    sep=' ',          # separador espacio
                    index=False,      # sin índice
                    header=False)
    elif fps==1:
        # In this case create the frame_list.csv file with the frame info at 1fps
        #TODO: add posibility to change this path depending if its original or cut_margin
        frames_path = '/media/SSD3/leoshared/Dataset/frames'
        fps1_dict = []

        #Use glob to load all possible frames at 1fps
        for video_idx in df['Video_name'].unique():
            video_frames_path = sorted(glob.glob(os.path.join(frames_path, video_idx, '*.jpg')))
            video_frames_path = video_frames_path[0:2701:30] #-> take only possitions that are multiples of 30

            for pth in video_frames_path:
                video_string = pth.split('/')[-2]
                frame_string = pth.split('/')[-1][:-4]
                
                fps1_dict.append({'Video_name': video_string,
                                'Video_id': int(video_string.split('_')[-1]),
                                'frame_id': int(frame_string),
                                'frame_path': os.path.join(video_string, f'{frame_string}.jpg')})
            
        fps1_df = pd.DataFrame(fps1_dict)
        fps1_df.to_csv(f'/home/dlmanrique/Endovis/CVS_Challenge/Models/MuST-Challenge-CVS-2025/data/Cvssages/only_challenge_annot_data/frame_lists/fold_{fold}_{split}_fps_{fps}.csv', 
                    sep=' ',          # separador espacio
                    index=False,      # sin índice
                    header=False )
    
    elif fps==30:
        # In this case create the frame_list.csv file with the frame info at 1fps
        #TODO: add posibility to change this path depending if its original or cut_margin
        frames_path = '/media/SSD3/leoshared/Dataset/frames'
        fps1_dict = []

        #Use glob to load all possible frames at 1fps
        for video_idx in df['Video_name'].unique():
            video_frames_path = sorted(glob.glob(os.path.join(frames_path, video_idx, '*.jpg')))

            for pth in video_frames_path:
                video_string = pth.split('/')[-2]
                frame_string = pth.split('/')[-1][:-4]
                
                fps1_dict.append({'Video_name': video_string,
                                'Video_id': int(video_string.split('_')[-1]),
                                'frame_id': int(frame_string),
                                'frame_path': os.path.join(video_string, f'{frame_string}.jpg')})
            
        fps1_df = pd.DataFrame(fps1_dict)
        fps1_df.to_csv(f'/home/dlmanrique/Endovis/CVS_Challenge/Models/MuST-Challenge-CVS-2025/data/Cvssages/only_challenge_annot_data/frame_lists/fold_{fold}_{split}_fps_{fps}.csv', 
                    sep=' ',          # separador espacio
                    index=False,      # sin índice
                    header=False )


def annotations_json_creator(df: pd.DataFrame, split: str, fold: int):
    """
    This function creates the annotations.json file with the info of the key frames and annotations
    """
    #Join all criterias in a single list
    df['cvs_annots'] = df[['c1_consensus', 'c2_consensus', 'c3_consensus']].values.tolist()
    df['Video_id'] = df['Video_name'].str.split('_').str[-1].astype(int)
    df['frame_path'] = df['Video_name'] + '/' + df['frame_id'].astype(str).str.zfill(5) + '.jpg'
    
    images_info = []
    annots_info = []
    base_frames_path = '/home/dlmanrique/Endovis/CVS_Challenge/Models/MuST-Challenge-CVS-2025/data/Cvssages/only_challenge_annot_data/frames'

    for index, row in tqdm(df.iterrows()):

        #Open each image and define width and height
        image_path = row['frame_path']
    
        if os.path.exists(os.path.join(base_frames_path, image_path)):
            with Image.open(os.path.join(base_frames_path, image_path)) as img:
                width, height = img.size
        else:
            print(f"Ruta no encontrada: {image_path}")
            width, height = None, None

        image_info = {'id': index,
                    'file_name': row['frame_path'],
                    'video_name': row['Video_name'],
                    'frame_num': index,
                    'width': width,
                    'height': height}
        
        annot_info = {'id': index,
                      'image_id': index,
                      'image_name': row['frame_path'],
                      'cvs': row['cvs_annots']}

        images_info.append(image_info)
        annots_info.append(annot_info)

    json_data = {'images':images_info, 'annotations':annots_info}

    save_json(json_data,
              f'/home/dlmanrique/Endovis/CVS_Challenge/Models/MuST-Challenge-CVS-2025/data/Cvssages/only_challenge_annot_data/annotations/{split}_long-term_fold_{fold}.json')



if __name__ == '__main__':

    #Open csv file with all the frames annotations info
    df_frames_annots = pd.read_csv('/media/SSD3/leoshared/Dataset/labels/unified_frame_labels.csv')

    #Open files with split info (for this only use the fold1 file, the fold2 is the same info just changing train for test)
    splits_info = load_json('/media/SSD3/leoshared/Dataset/Splits_partition/fold1_video_splits.json')
    train_videos, test_videos = splits_info['train'], splits_info['test']
    #Operate the c_i columns and apply mayority vote consensus
    df_frames_annots = majority_vote_consensus(df_frames_annots)
    #Remove unnecesary columns
    df_frames_annots = df_frames_annots[['Video_name', 'frame_id', 'c1_consensus', 'c2_consensus', 'c3_consensus']]

    # Fold1 configurations and files
    df_train_fold1 = df_frames_annots[df_frames_annots['Video_name'].isin(train_videos)]
    df_test_fold1 = df_frames_annots[df_frames_annots['Video_name'].isin(test_videos)]

    print('\n Processing files for fold 1....')
    for split, df in zip(['train', 'test'], [df_train_fold1, df_test_fold1]):
        frame_list_creator(df, split, 1, fps=30)
        annotations_json_creator(df, split, 1)

    print('\n Processing files for fold 2....')
    # Fold2 configurations and files
    #TODO: ojo con esto, aca solo cambio las listas pero es muy susceptible a errores humanos
    df_train_fold2 = df_frames_annots[df_frames_annots['Video_name'].isin(test_videos)] 
    df_test_fold2 = df_frames_annots[df_frames_annots['Video_name'].isin(train_videos)]

    for split, df in zip(['train', 'test'], [df_train_fold2, df_test_fold2]):
        frame_list_creator(df, split, 2, fps=30)
        annotations_json_creator(df, split, 2)





    
