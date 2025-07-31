#This file create teh json files that follows the 80-20 splitting for training MuST as
# method for extending annotations

import json
import os
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image

random.seed(42)

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
    base_frames_path = '/home/dlmanrique/Endovis/CVS_Challenge/Models/MuST-Challenge-CVS-2025/data/Cvssages/extend_annots/frames'

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
              f'/home/dlmanrique/Endovis/CVS_Challenge/Models/MuST-Challenge-CVS-2025/data/Cvssages/extend_annots/annotations/{split}_long-term_extend_annots.json')




if __name__ =="__main__":

    #Open csv file with all the frames annotations info
    df_frames_annots = pd.read_csv('/media/SSD3/leoshared/Dataset/labels/unified_frame_labels.csv')

    #Operate the c_i columns and apply mayority vote consensus
    df_frames_annots = majority_vote_consensus(df_frames_annots)
    #Remove unnecesary columns
    df_frames_annots = df_frames_annots[['Video_name', 'frame_id', 'c1_consensus', 'c2_consensus', 'c3_consensus']]

    train_videos_ids = random.sample(range(1, 701), 560)
    all = set(range(1, 701))
    test_videos_ids = sorted(list(all - set(train_videos_ids)))

    train_videos = [f'video_{str(idx).zfill(3)}' for idx in train_videos_ids]
    test_videos = [f'video_{str(idx).zfill(3)}' for idx in test_videos_ids]

    df_train = df_frames_annots[df_frames_annots['Video_name'].isin(train_videos)]
    df_test = df_frames_annots[df_frames_annots['Video_name'].isin(test_videos)]

    for split, df in zip(['train', 'test'], [df_train, df_test]):
        annotations_json_creator(df, split, 1)