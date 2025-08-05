
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict



def frames_folder_creator(split: str):

    source_dir = Path(f'/media/lambda001/SSD3/leoshared/Dataset/endoscapes/{split}')  # donde están los archivos 1_12457.jpg
    target_dir = Path('/media/SSD4/dlmanrique/Endovis/CVS_Challenge_2025/MuST-Challenge-CVS-2025/data/Endoscapes/frames')   # donde quieres los symlinks
   
    video_frames = defaultdict(list)

    for file in source_dir.glob('*.jpg'):
        filename = file.stem  # e.g., '1_12457'
        video_id, frame_id = filename.split('_')
        video_frames[video_id].append((int(frame_id), file))  # guarda frame_id como int para ordenar

    # Procesar cada grupo de frames
    for video_id, frames in tqdm(video_frames.items()):
        video_name = f'video_{int(video_id):03d}'  # e.g., 'video_001'
        output_folder = target_dir / video_name
        output_folder.mkdir(parents=True, exist_ok=True)

        # Ordenar frames por frame_id
        sorted_frames = sorted(frames, key=lambda x: x[0])

        for new_idx, (frame_id, file) in enumerate(sorted_frames, start=1):
            new_name = f"{new_idx:05d}.jpg"  # e.g., '00001.jpg'n
            symlink_path = output_folder / new_name
            if not symlink_path.exists():
                symlink_path.symlink_to(file.resolve())


if __name__ == "__main__":

    for split in ['train', 'val', 'test']:
        print(f'Processing frames for split {split}')
        frames_folder_creator(split)