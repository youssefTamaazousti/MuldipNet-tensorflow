# MulDiPNet: Multi Discriminative Problem Network

## Introduction
Code for the [CVPR'17](http://perso.ecp.fr/~tamaazouy/files/pdf/MuCaLe_Net_Multi_Categorical_Level_Networks_to_Generate_More_Discriminating_Features.pdf) (MuCaLe-Net: Multi Categorical-Level Networks to Generate More Discriminating Features) and [ArXiv:2120918](http://perso.ecp.fr/~tamaazouy/files/pdf/Learning_More_Universal_Representations_for_Transfer_Learning.pdf) (Learning More Universal Representations for Transfer-Learning) papers.

## Source and Target-Datasets Preparation
Roughly, here we convert the raw images and their associated labels to the tfrecord format [(see here for detailed explanation)](todo:link). We also compute the mean-image on the training set and store their mean values (for each RGB channel) in a file. 

To do so, simply run the following commands (with the first command converting the source-datasets and the second, converting the target-ones):
```
python dataset.source_task.tfrecord.py
python dataset.target_task.tfrecord.py
```
This may take a while (especially for the source-task that contain a very large amount of images). 
Note that, the dataset lists have to be in the following format (path to images and label): 

```
/path/to/raw/images/name_image_1.jpg 0
/path/to/raw/images/name_image_1.jpg 1
```

in progress...

## Networks Training

in progress...

## Features Extraction

in progress...

## Citations
If you find the codes useful, please cite these papers:
```
@article{tamaazousti2018universal,
  title={Learning More Universal Representations for Transfer-Learning},
  author={Tamaazousti, Youssef and Le Borgne, Herv\'e and Hudelot, C\'eline and Seddik, Mohamed El Amine and Tamaazousti, Mohamed},
  journal={arXiv:1712.09708},
  year={2017}
} 
```
and 
```
@inproceedings{tamaazousti2017mucale_net,
  title={MuCaLe-Net: Multi Categorical-Level Networks to Generate More Discriminating Features},
  author={Tamaazousti, Youssef and Le Borgne, Herv{\'e} and Hudelot, C{\'e}line},
  booktitle={Computer Vision and Pattern Recognition},
  series={CVPR},
  year={2017}
} 
```
