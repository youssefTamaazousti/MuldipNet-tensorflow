# MulDiPNet: Multi Discriminative Problem Network

## Introduction
Code for the [CVPR'17](http://perso.ecp.fr/~tamaazouy/files/pdf/MuCaLe_Net_Multi_Categorical_Level_Networks_to_Generate_More_Discriminating_Features.pdf) (MuCaLe-Net: Multi Categorical-Level Networks to Generate More Discriminating Features) and [ArXiv:1712.09708](https://arxiv.org/pdf/1712.09708) (Learning More Universal Representations for Transfer-Learning) papers. 
You can use this code with Tensorflow (gpu version) compiled. 

This code includes:
- Code to convert datasets into Tensorflow [tfrecord format](https://www.tensorflow.org/programmers_guide/datasets) (it speeds-up the read of data and thus the training process). The python program  `convert2tfrecord.task.py` (with `task` being `source_task` or `target_task`) does this.
- Code to train MulDiPNet on an arbitrary deep convolutional neural network provided in the [Slim format](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) of Tensorflow. This is performed by the python program `train_network.source_task.architecture.py` (with `architecture` being `alexnet` or `darknet`).
- Code to extract features from any target-dataset (that has to be converted in tfrecord format) through a pre-trained MulDiPNet. This is performed by the python program `extract_features.architecture.py` (with `architecture` being `alexnet` or `darknet`).
- Code to perform a SPV (Source Problem Variation) given an initial source-problem (SP) (*i.e.*, list of images and their associated integer-labels with a file containing each integer-label associated to a certain word-label), and a hierarchy on which are mapped the word-labels of the initial SP. To do so, you have to run the `spv.py` program. 

If you want to work on cpu, remove all the `with tf.device('/gpu:'+str(cfg.GPU)):` lines from the code. 

## Converting (Source and Target) Datasets to Tfrecord Format
Roughly, here we convert the raw images and their associated labels to the tfrecord format [(see here for detailed explanation)](todo:link). We also compute the mean-image on the training set and store their mean values (for each RGB channel) in a file. 

To do so, simply run the following commands (with the first command converting the source-datasets and the second, converting the target-ones):
```
python dataset.source_task.tfrecord.py
python dataset.target_task.tfrecord.py
```
This may take a while (especially for the source-task that contain a very large amount of images). 
Note that, the dataset lists have to be in the following format: two columns with the first column contains image-names with their absolute paths and second column contains their associated label. 
An exemple of such a format is presented in the following:

```
/path/to/raw/images/name_image_1.jpg 0
/path/to/raw/images/name_image_1.jpg 1
```

All the lists for each of the source-problem ILSVRC-half of the paper are included in the `data/ilsvrc_half/lists` folder.

## Training MulDiP-Net

in progress...

## Extract Features Through Pre-Trained MulDiP-Net

in progress...

## Perform SPV (Source Problem Variation) 

in progress...

## Citation
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
