# MuST: Multi-Scale Transformers for Surgical Phase Recognition


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/must-multi-scale-transformers-for-surgical/surgical-phase-recognition-on-grasp)](https://paperswithcode.com/sota/surgical-phase-recognition-on-grasp?p=must-multi-scale-transformers-for-surgical) 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/must-multi-scale-transformers-for-surgical/surgical-phase-recognition-on-heichole)](https://paperswithcode.com/sota/surgical-phase-recognition-on-heichole?p=must-multi-scale-transformers-for-surgical) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/must-multi-scale-transformers-for-surgical/surgical-phase-recognition-on-misaw)](https://paperswithcode.com/sota/surgical-phase-recognition-on-misaw?p=must-multi-scale-transformers-for-surgical) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/must-multi-scale-transformers-for-surgical/surgical-phase-recognition-on-cholec80-1)](https://paperswithcode.com/sota/surgical-phase-recognition-on-cholec80-1?p=must-multi-scale-transformers-for-surgical)

<div align="center">
  <h3>MuST Architecture</h3>
  <img src="/src/MuST_architecture.png" alt="MuST Architecture"/>
  <br/><br/>
  <h3>Multi-Term Frame Encoder (MTFE) Architecture</h3>
  <img src="/src/MTFE_architecture.png" alt="MTFE Architecture"/>
</div>
<br/>


# Model Description

We present Multi-Scale Transformers for Surgical Phase Recognition (MuST), a two-stage Transformer-based architecture designed to enhance the modeling of short-, mid-, and long-term information within surgical phases. Our method employs a frame encoder that leverages multi-scale surgical context across different temporal dimensions. The frame encoder considers diverse time spans around a specific frame of interest, which we call a keyframe. The keyframe serves as the specific frame that we encode. We construct temporal windows around this keyframe to provide the necessary temporal context for accurate phase classification. Our encoder generates rich embeddings that capture short- and mid-term dependencies. To further enhance long-term understanding, we employ a Temporal Consistency Module that establishes relationships among frame embeddings within an extensive temporal window, ensuring coherent phase recognition within an extensive temporal window.

- **Confernece paper** in Medical Image Computing and Computer Assisted Intervention – **MICCAI 2024**. Proceedings available at [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72089-5_40)

- **Preprint** available at [Arxiv](https://arxiv.org/pdf/2407.17361)

- **Winning solution** of the [2024 PhaKIR Challenge](https://phakir.re-mic.de/)

- You can also visit our [**Project Page**](https://dioses-miccai2024.github.io/must_page/#)

## Installation
Please follow these steps to run MuST:

```sh
$ conda create --name must python=3.8 -y
$ conda activate must
$ conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

$ conda install av -c conda-forge
$ pip install -U iopath
$ pip install -U opencv-python
$ pip install -U pycocotools
$ pip install 'git+https://github.com/facebookresearch/fvcore'
$ pip install 'git+https://github.com/facebookresearch/fairscale'
$ python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

$ git clone https://github.com/BCV-Uniandes/MuST
$ cd MuST
$ pip install -r requirements.txt
```

## Data Preparation

The [DATA_PREPARATION.md](DATA_PREPARATION.md) file contains detailed instructions for preparing the datasets used to validate our method, downloading pre-trained model weights, and guidelines for setting up your own custom dataset.

## Running the code

| Dataset   | Test Metric (metric) | Config          | Run File          | Frames Features   | Model           |
|-----------|----------------------|-----------------|-------------------|-------------------|-----------------|
| GraSP     | 79.14 (mAP)      | [GrasP TCM Config](/configs/GraSP/TCM_PHASES.yaml)       | [Run GraSP TCM](/run_files/tcm/grasp_phases.sh) | ./data/GraSP/frames_features  | [GrasP Weights](https://drive.google.com/drive/folders/1gILc24qhrxsbRxPtq6pz7qyOYFxBCsZY?usp=drive_link)    |
| MISAW     | 98.08 (mAP)      | [MISAW TCM Config](/configs/misaw/TCM_PHASES.yaml)       | [Run MISAW TCM](/run_files/tcm/misaw_phases.sh) | ./data/misaw/frames_features  | [MISAW Weights](https://drive.google.com/drive/folders/15q3Y4Uo0H9Z4MBOkqxCwSiJL5QGoHc-t?usp=drive_link)     |
| HeiChole  | 77.25 (F1-score) | [HeiChole TCM Config](/configs/heichole/TCM_PHASES.yaml) | [Run HeiChole TCM](/run_files/tcm/heichole_phases.sh) |./data/heichole/frames_features  | [Heichole Weights](https://drive.google.com/drive/folders/1m8HMCwmaGjEyTGgCSAiTgwV0T0r0CVSJ?usp=drive_link) |
| Cholec80  | 85.57 (F1-score) | [Cholec80 TCM Config](/configs/cholec80/TCM_PHASES.yaml) | [Run Cholec80 TCM](/run_files/tcm/cholec80_phases.sh) | ./data/cholec80/frames_features |[Cholec80 Weights](https://drive.google.com/drive/folders/1-n_UUNMSXYH2E4jXMBlR44pYvzwm5-EL?usp=drive_link)     |

We provide bash scripts with the default parameters to evaluate each dataset. Please first download our preprocessed data files and pretrained models as instructed earlier and run the following commands to run evaluation on each task:

```sh
# Calculate features running the script corresponding to the desired dataset
$ sh run_files/extract_features/{dataset}_phases
# Run the script corresponding to the desired dataset to evaluate
$ sh run_files/tcm/{dataset}_phases
```

### Training MuST

You can easily modify the bash scripts to train our models. Just set ```TRAIN.ENABLE True``` on the desired script to enable training, and set ```TEST.ENABLE False``` to avoid testing before training. You might also want to modify ```TRAIN.CHECKPOINT_FILE_PATH``` to the model weights you want to use as initialization. You can modify the [config files](configs/) or the [bash scripts](run_files/) to modify the architecture design, training schedule, video input design, etc. We provide documentation for each hyperparameter in the [defaults script](./must/config/defaults.py). For the Temporal Consistency Module (TCM), ensure the temporal chunks are being used by setting ```TEMPORAL_MODULE.CHUNKS True```. For more details to train MuST, refer to [TRAINING.md](TRAINING.md)


## Citation

If you find this repository helpful, please consider citing:

```BibTeX

@inproceedings{perez2024must,
  title={MuST: Multi-scale Transformers for Surgical Phase Recognition},
  author={P{\'e}rez, Alejandra and Rodr{\'\i}guez, Santiago and Ayobi, Nicol{\'a}s and Aparicio, Nicol{\'a}s and Dessevres, Eug{\'e}nie and Arbel{\'a}ez, Pablo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={422--432},
  year={2024},
  organization={Springer}
}

```
