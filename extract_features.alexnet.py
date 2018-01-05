from __future__ import division
import numpy as np
import tensorflow as tf
import datetime
import os
import argparse
from PIL import Image
from glob import glob
from read_data_list import READ_DATA_LIST
from time import gmtime, strftime

from alexnet_slim_extraction import ALEXNET

def main():
 
 ########################
 #      Parameters      #
 ########################
 # Declare argument-parser 
 parser = argparse.ArgumentParser()
 parser.add_argument('--architecture', type=str, default='alexnet', help='Name of the network-architecture to use')
 parser.add_argument('--source_task_dataset', type=str, default='ilsvrc_half', help='Name of the dataset used for the pre-trained network')
 parser.add_argument('--source_task_SP', type=str, nargs='+', default=['_S', '_S_GroupingCategoricalLevels'], help='Name of the source-problem (SP) used for the pre-trained network')
 parser.add_argument('--model_iter', type=int, default=500000, help='Number of iterations on which the source-task model was saved')
 parser.add_argument('--target_task_dataset', type=str, nargs='+', default=['voc07', 'voc12'], help='Name of the target-datasets on which we want to extract features')
 parser.add_argument('--target_task_phase', type=str, nargs='+', default=['train', 'test'], help='Name of the target-datasets on which we want to extract features')
 parser.add_argument('--layer2extract', type=str, default='fc_7', help='Name of the layer of the network to extract')
 parser.add_argument('--gpu', type=int, default=0, help='GPU device on which are declared the variables')
 # TODO:
 #source_dataset > source_task_dataset
 #source_dataset_name > source_task_SP
 #weights_iter > model_iter
 #target_dataset > target_task_dataset
 #phase > target_task_phase
 #layer_to_extract > layer2extract
 #gpu
 
 args = parser.parse_args()
 # Get the network-architecture from the arguments (if no argument, alexnet is used by default) 
 if args.architecture=="alexnet":
    from architecture_alexnet_forTraining import NET
 elif args.architecture=="darknet":
    from architecture_darknet_forTraining import NET
 print("architecture = "+str(args.architecture))
 
 # Parameters of pre-trained nework
 source_task_dataset = args.source_task_dataset #'ilsvrc_half' 
 source_task_SP = args.source_task_SP #'_S_GroupingCategoricalLevels' #'_S_GroupingCategoricalLevels' #'_S_GroupingHierarchicalLevels5' #'_S_Adding100' #'_S_GroupingCategoricalLevels' #'_S_Removing241' #'_S' #'_S_GroupingHierarchicalLevels7'
 source_dataset_name_spec = '_RandomCropAndFlip'
 training_strategy = '' #'_FS' FS means training From Scratch #'_FineTuning_FC78_2k'
 weights_iter = 500000 
 batch_size = 1
 
 with tf.device('/gpu:'+str(args.gpu)):
     layer_to_extract = 'fc_7' 
     nbr_classes = 577
     data_transformation = 1
     # Declare network (AlexNet)
     network = NET(name_layer2extract=layer_to_extract, output_size=nbr_classes)
     variable_to_restore = tf.global_variables() # for finetuning on same database
     saver4restoring = tf.train.Saver(variable_to_restore, max_to_keep=None) # saver for restoring variable_to_restore
     global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
     
 for target_dataset in ['voc07', 'voc12', 'mit67', 'ca101', 'ca256', 'cub200', 'nwOB', 'stACT', 'stCAR', 'fl102']: #['stACT', 'fl102', 'stCAR', 'voc12', 'ca256']: #['voc07', 'mit67', 'ca101', 'ca256', 'cub200', 'nwOB', 'voc12', 'stCAR']: # ilsvrc_half
  for phase in ['train', 'test']:
    # TODO: loop for source_task_SP: for source_task_SP in args.source_task_SP:
    print ('Extracting features for '+phase+' phase of '+target_dataset+' dataset')

    #######################
    # Inferred Parameters #
    #######################
    # Paths
    path_to_source_data = '/scratch_global/youssef/expes/gs_complementarity/transfer_learning/' + source_dataset +'/'
    path_to_target_data = '/scratch_global/youssef/expes/gs_complementarity/transfer_learning/' + target_dataset +'/'
    weights_file = path_to_source_data + 'modelss'+source_dataset_name+source_dataset_name_spec+'/model.'+source_dataset+source_dataset_name+'.config_#classes='+str(nbr_classes)+'_imgSize=227_imgSpec='+source_dataset_name_spec[1:]+training_strategy+'.ckpt-'
    weights_file += str(weights_iter)
    output_file_name = path_to_target_data + 'features_TF/'+target_dataset+'.'+phase+'.FS.'+layer_to_extract+'.cnn'+source_dataset_name+source_dataset_name_spec+training_strategy+'.alexnet_full.'+str(int(weights_iter/1000))+'kIter.txt'
    # Amount of images in target-dataset
    list_filename = path_to_target_data + target_dataset + '.' + phase + '.lst'
    with open(list_filename, 'r') as list_file:
        list_image = [x.strip() for x in list_file.readlines()]
    nbr_total_img = int(len(list_image))
    
    #######################
    # Reading mean-values #
    #######################
    mean_filename = path_to_source_data+ 'mean_images/mean_image.'+ source_dataset + source_dataset_name +'.txt'
    with open(mean_filename, 'r') as mean_file:
       mean_image = [x.strip() for x in mean_file.readlines()]

    ###########################
    # Reading tf-records file #
    ###########################
    tfrecords_filename = path_to_target_data + 'tfrecord_files/tfrecord_files'+ target_dataset + '.' + phase + '.bin' 
    filename_queue = tf.train.string_input_producer([tfrecords_filename]) #, num_epochs=1) #shuffle=False) 
    feature = {
       'height': tf.FixedLenFeature([], tf.int64, default_value=-1),
       'width': tf.FixedLenFeature([], tf.int64, default_value=-1),
       'image_raw': tf.FixedLenFeature((), tf.string, default_value='')
    }

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)

    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    image = tf.image.resize_images (image, [256, 256])
    if data_transformation == 1:
       # random crop and flip
       image = tf.random_crop(image, [227, 227, 3])
       image = tf.image.random_flip_left_right(image)
    else: 
       # center crop 
       image = tf.image.resize_image_with_crop_or_pad(image, 227, 227)
    mean = tf.constant([float(mean_image[0]), float(mean_image[1]), float(mean_image[2])], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 1, 3])
    image = image - mean

    images = image #tf.train.batch([image], batch_size=1, capacity=30, num_threads=1)

    with tf.device('/gpu:'+str(args.gpu)):        
     ######################
     # Declare parameters #
     ###################### 
     """
     variable_to_restore = tf.global_variables() # for finetuning on same database
     saver4restoring = tf.train.Saver(variable_to_restore, max_to_keep=None) # saver for restoring variable_to_restore
     global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
     """
    with tf.device('/gpu:'+str(args.gpu)):
     #global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

     sess = tf.InteractiveSession()
     tf.global_variables_initializer().run()
     
     # Restoring weights from pre-trained model
     if weights_file is not None:
        print('\nRestoring weights from: ' + weights_file + '\n\n')
        saver4restoring.restore(sess, weights_file)

    ##############
    # Extraction #
    ##############
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    output_file = open(output_file_name, 'w')
    print('Total images: ' + str(nbr_total_img))
    nbr_batches = nbr_total_img 
    for step in range(nbr_batches): 
       # Display progress
       if step%1000 == 0:
          print(str(step)+'/'+str(nbr_batches))

       # Extraction
       test_images = sess.run([images])
       feed_dict_test = {network.images: test_images, network.keep_prob: 1.0}
       output_descriptor = np.asarray(sess.run(network.logits, feed_dict=feed_dict_test))
       #print(output_descriptor.shape)
       
       #####################################
       # Writing descriptors in ASCII file #
       #####################################
       # convolutional/pooling features (TODO: flatten of matrices)

       # fully-connected features (vectors)
       for i in range(0, output_descriptor.shape[1]):
          if str(output_descriptor[0, i]) == '0.0':
             output_file.write('0 ')
          else:
             output_file.write(str(output_descriptor[0, i])+' ')
       output_file.write('\n')

    output_file.close()
    # Stop the threads
    coord.request_stop()
    # Wait for threads to stop
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()

