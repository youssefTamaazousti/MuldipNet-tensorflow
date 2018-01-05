import os

###################
# Data parameters #
###################
DATASET = 'ilsvrc_half' #'ilsvrc_full'  
SOURCE_PROBLEM = '_S' #'_S_GroupingCategoricalLevels'  # TODO: list of SP in order to train them all one after the other
IMAGES_SPEC = 'RandomCropAndFlip' 
#TRAINING_STRATEGY = 'FromScratch' # TODO: 'standard'
NUMBER_CATEGORIES = 483 #200 # TODO:  list in the same order than the list of SPs
IMAGE_RESIZE = 256
IMAGE_SIZE = 227 
WEIGHTS_FILE = None 
# if you want to start from a particular checkpoint, comment the previous line and uncomment the next one (also set the correct path to your model)
#WEIGHTS_FILE = "path/to/your/trained/model.ckpt-500000" # here 500000 corresponds to the number of iterations of the pre-trained model

####################
# Solver parameters
#################### 
DISPLAY_ITER = 100 
MAX_ITER = 500000 
SAVE_ITER = 10000  

LEARNING_RATE = 0.01
STAIRCASE = True
DECAY_STEPS = 100000 # decrease learning-rate every DECAY_STEPS iterations
DECAY_RATE = 0.1 # factor of decrease of learning-rate

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
INITIAL_WEIGHTS_MEAN = 0.0
INITIAL_WEIGHTS_STD = 0.01
INITIAL_BIAS = 1.0


