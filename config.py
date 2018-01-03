import os

#############################
# Data parameters (TO CHANGE)
##############################
DATASET_NAME = 'ilsvrc_half' #'ilsvrc_half' #'mit67' 'voc07' 
DATASET_SPEC = '_S_AddingGeneric100' #'_S' #'_S_GroupingCategoricalLevels' #'_S_Adding100' #'_S_Removing241'  
IMAGES_SPEC = 'RandomCropAndFlip' #'RandomCrop' # 'CenterCrop'
#TRAINING_STRATEGY = #'SFT_FC78_4k' #'PreTraining_G' #'PreTraining_G' #'FineTuning_FC78_2k' #'FineTuning_FC78_2k' #'FineTuning_FC678_FC78' #'FromScratch'
NC = 577 #483 #24 #63 #100 #148 #194 #241 #483 #200 #583 #242 
GPU = '2' 
BATCH_SIZE = 256
ITER_MODEL = 310000 #500000


#############
# File paths
#############
IMAGE_RESIZE = 256
IMAGE_SIZE = 227
DATA_PATH = '/scratch_global/youssef/expes/gs_complementarity/transfer_learning' 
DATASET_PATH = os.path.join(DATA_PATH, DATASET_NAME)
CACHE_PATH = os.path.join(DATASET_PATH, 'cache')
OUTPUT_DIR = os.path.join(DATASET_PATH, 'modelss'+DATASET_SPEC+'_'+IMAGES_SPEC)
SAVE_FILE_NAME = 'model.'+str(DATASET_NAME)+str(DATASET_SPEC)+'.config_#classes='+str(NC)+'_imgSize='+str(IMAGE_SIZE)+'_imgSpec='+str(IMAGES_SPEC)+'.ckpt'
#SAVE_FILE_NAME = 'model.'+str(DATASET_NAME)+str(DATASET_SPEC)+'.config_#classes='+str(NC)+'_imgSize='+str(IMAGE_SIZE)+'_imgSpec='+str(IMAGES_SPEC)+'_'+TRAINING_STRATEGY+'.ckpt'

WEIGHTS_FILE = None
#WEIGHTS_FILE = os.path.join(DATASET_PATH, OUTPUT_DIR, 'model.'+str(DATASET_NAME+DATASET_SPEC)+'.config_#classes='+str(NC)+'_imgSize=227_imgSpec='+str(IMAGES_SPEC)+'_'+TRAINING_STRATEGY+'.ckpt-'+str(ITER_MODEL))  
#WEIGHTS_FILE = os.path.join(DATASET_PATH, os.path.join(DATASET_PATH, 'modelss_S_GroupingCategoricalLevels_'+IMAGES_SPEC), 'model.'+str(DATASET_NAME+'_S_GroupingCategoricalLevels')+'.config_#classes='+str(200)+'_imgSize=227_imgSpec='+str(IMAGES_SPEC)+'.ckpt-'+str(ITER_MODEL))
#WEIGHTS_FILE = "/scratch_global/youssef/expes/gs_complementarity/training_source_task/YOLO_small.ckpt"

####################
# Solver parameters
#################### 
DISPLAY_ITER = 100 
MAX_ITER = 500000 #60000 # TODO: 500000
TEST_ITER = 5000
SAVE_ITER = 10000  

LEARNING_RATE = 0.01 #0.001 # TODO: 0.01 
LEARNING_RATE_PRETRAINED = 0.0001 # TODO: LEARNING_RATE/10
DECAY_STEPS = 100000 #20000 #TODO: 100000 # decrease learning-rate every DECAY_STEPS iterations
DECAY_RATE = 0.1 # factor of deacrease of learning-rate
STAIRCASE = True  

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
INITIAL_WEIGHTS_MEAN = 0.0
INITIAL_WEIGHTS_STD = 0.01
INITIAL_BIAS = 1.0


