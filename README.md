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
python src/dataset.source_tasks.tfrecord.py
python src/dataset.target_tasks.tfrecord.py
```
This may take a while (especially for the source-task that contain a very large amount of images). 
Anyway, it will output a tfrecord file (*e.g.*, `tfrecord_file.task_name.bin`) for each dataset that will be located in `data/tfrecord_files/`. For the source-task, it will also output a mean-values file (*e.g.*, `mean_values.source_task.txt`) located in `data/mean_values/`, containing the mean values (for each RGB channel) of the mean image computed on the whole training set.  
Note that, the dataset lists have to be in the following format: two columns with the first column contains image-names with their absolute paths and second column contains their associated label. 
An exemple of such a format is presented in the following:

```
/path/to/raw/images/name_image_1.jpg 0
/path/to/raw/images/name_image_1.jpg 1
```
All the lists for each of the source-problem ILSVRC-half of the paper are included in the `data/ilsvrc_half/lists` folder.

## Training MulDiP-Net

Once the source-tasks data are prepared and converted in tfrecord, we can train the MulDiP-Net, for each source-task as follow:
```
python src/train_network.alexnet.source_tasks.py \
  --architecture darknet \
  --batch_size 256 \
  --gpu 0
```
The latter command has to be run for each source-problem. To set the source-problems as well as other parameters, you should modify the following config file: `utils/config.py` .
During training, the latter program will save a model every `SAVE_ITER` (see config file) iterations and will print the loss and accuracy on a training-batch every `DISP_ITER` iterations. 

## Extract Features Through Pre-Trained MulDiP-Net

Once the MulDiP-Net is trained on the source-tasks, we can use it to extract features on target-tasks as follow: 
```
python src/extract_features.target_tasks.py \
  --architecture alexnet \
  --source_task_dataset ilsvrc_half \
  --source_task_SP SP_INIT SPV_G_CAT SPV_G_HIE SPV_CLU \
  --model_iter 500000 \
  --target_task_dataset voc07 voc12 \
  --target_task_phase train test \
  --layer2extract fc_7 \
  --gpu 0
```
Here with the source-tasks being `SP_INIT SPV_G_CAT SPV_G_HIE SPV_CLU`, the latter code will output a set of four feature files (i.e., as many files as the amount of source-problems considered). 
Each features file contain N lines and D columns, with N being the amoung of images in your dataset (*e.g.*, `N=5011` for the training set of VOC 2007) and D being the dimensionality of the layer extracted (*e.g.*, `D=4096` for the fc7 layer of AlexNet).
Then, these features are independently normalized and combined with the following:
```
python src/combine_normalize_features.target_tasks.py \
  --architecture alexnet \
  --source_task_dataset ilsvrc_half \
  --source_task_SP SP_INIT SPV_G_CAT SPV_G_HIE SPV_CLU \
  --model_iter 500000 \
  --target_task_dataset voc07 voc12 \
  --target_task_phase train test \
  --normalization Linf \
  --combination concat
```
This latter will take as input the set of four feature files and output one features file containing the normalized and combined features.

## Perform Grouping-SPV (Source Problem Variation) 

To perform Grouping-SPV, we first need to have a hierarchy on which are mapped the word-labels of the initial SP (in the first column of a file) and the word-labels of the final SP (in the second column). 
in progress...

# Categorical-levels
The `MulDiPNet/data/concept_levels` folder contains three files:
- `concepts_subordinate-level_ILSVRC1000.txt` : It contains the set of 1000 categories from the subordinate-level of ILSVRC 
- `concepts_basic-level_ILSVRC1000.txt` : It contains the set of 682 categories from the basic-level of ILSVRC
- `association_synsets_subordinate-level_to_synsets_basic-level_ILSVRC1000.txt` : It contains the association of the 1000 subordinate-categories of ILSVRC (first column of the file) with the 682 basic-level categories of ILSVRC (second column of the file). Note that, in the second column, when the synset-id is printed, it means that no basic-level exist (and it is actually an entry-level category). 

# Hierarchical-levels
in progress...


# Clustering-levels
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
