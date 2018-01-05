import os

###################
# Data parameters #
###################
DATASET_NAME = 'ilsvrc_half' #'ilsvrc_full'  
DATASET_SPEC = '_S' #'_S_GroupingCategoricalLevels'  
IMAGES_SPEC = 'RandomCropAndFlip' 
#TRAINING_STRATEGY = #'SFT_FC78_4k' #'PreTraining_G' #'PreTraining_G' #'FineTuning_FC78_2k' #'FineTuning_FC78_2k' #'FineTuning_FC678_FC78' #'FromScratch'
NC = 577 #483 #24 #63 #100 #148 #194 #241 #483 #200 #583 #242 
IMAGE_RESIZE = 256
IMAGE_SIZE = 227
GPU = '2' 
BATCH_SIZE = 256
ITER_MODEL = 500000

################
#  File-paths  #
################
DATA_PATH = '/data' 
DATASET_PATH = os.path.join(DATA_PATH, DATASET_NAME)
OUTPUT_DIR = os.path.join(DATASET_PATH, 'models'+DATASET_SPEC+'_'+IMAGES_SPEC)
SAVE_FILE_NAME = 'model.'+str(DATASET_NAME)+str(DATASET_SPEC)+'.config_#classes='+str(NC)+'_imgSize='+str(IMAGE_SIZE)+'_imgSpec='+str(IMAGES_SPEC)+'.ckpt'
WEIGHTS_FILE = None

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


