import numpy as np
import tensorflow as tf

import datetime
import os
import argparse

from PIL import Image
from glob import glob

import config as cfg
from alexnet_slim import ALEXNET
from dataset import DATASET
from time import gmtime, strftime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--gpu', type=str)
    args = parser.parse_args()


    ##################
    #   Parameters   #
    ##################
    dataset_name = cfg.DATASET_NAME
    dataset_spec = cfg.DATASET_SPEC
    dataset_image_spec = cfg.IMAGES_SPEC
    path_training_source_task = '/scratch_global/youssef/expes/gs_complementarity/training_source_task/'
    path_data = '/scratch_global/youssef/expes/gs_complementarity/transfer_learning/'+dataset_name+'/'


    display_iter = cfg.DISPLAY_ITER
    max_iter = cfg.MAX_ITER
    test_iter = cfg.TEST_ITER
    save_iter = cfg.SAVE_ITER
    batch_size = cfg.BATCH_SIZE
    weights_file = cfg.WEIGHTS_FILE
    image_resize = cfg.IMAGE_RESIZE
    image_size = cfg.IMAGE_SIZE
    initial_learning_rate = cfg.LEARNING_RATE
    decay_steps = cfg.DECAY_STEPS
    decay_rate = cfg.DECAY_RATE
    staircase = cfg.STAIRCASE
    momentum = cfg.MOMENTUM
    nc = cfg.NC
    save_file_name = cfg.SAVE_FILE_NAME

    # Reading mean-values
    mean_values_filename = path_data + 'mean_images/mean_image.'+dataset_name + dataset_spec  +'.txt'
    with open(mean_values_filename, 'r') as mean_values_file:
       mean_values = [x.strip() for x in mean_values_file.readlines()]  
    R_mean = float(mean_values[0])
    G_mean = float(mean_values[1])
    B_mean = float(mean_values[2])

    ###########################
    # READING TF-RECORD files #
    ###########################
    tfrecords_filename = path_data +'tfrecord_files/tfrecord_files'+dataset_spec +'_'+dataset_image_spec +'.train.bin'  
    filename_queue = tf.train.string_input_producer([tfrecords_filename]) #, num_epochs=90) #shuffle=True)

    feature = {
       'height': tf.FixedLenFeature([], tf.int64, default_value=-1),
       'width': tf.FixedLenFeature([], tf.int64, default_value=-1),
       'image_raw': tf.FixedLenFeature((), tf.string, default_value=''), #[], tf.string),
       'annotation_raw': tf.FixedLenFeature([], tf.int64, default_value=-1)
    }
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)

    image = tf.image.decode_jpeg(features['image_raw'], channels=3)

    # Cast label data into int32
    label = tf.cast(features['annotation_raw'], tf.int64)

    #a = features['annotation_raw']

    # PREPROCESSING IMAGE
    image = tf.image.resize_images (image, [image_resize, image_resize])
    image = tf.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    mean = tf.constant([R_mean, G_mean, B_mean], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 1, 3])
    image = image - mean
    
    label_to_feed = tf.one_hot(label, nc, on_value=1, off_value=0)
    label_to_feed = tf.cast(label_to_feed, tf.float32)

    # Creates batches by randomly shuffling tensors
    images, annotations = tf.train.batch([image, label_to_feed], batch_size=batch_size, capacity=30, num_threads=3) #capacity=30, num_threads=12)
    
    with tf.device('/gpu:'+str(cfg.GPU)):
     # Declare alexnet-network
     network = ALEXNET()

     #############################
     # Declare solver-parameters #
     #############################
     variable_to_restore = tf.global_variables() # for finetuning on same database
     #variable_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=["alexnet/fc_6", "alexnet/fc_7"])

     # Declare saver for restoring variable_to_restore
     saver4restoring = tf.train.Saver(variable_to_restore, max_to_keep=None) # saver for restoring variable_to_restore
   
    # Declare saver for saving all variables 
    saver4saving = tf.train.Saver(max_to_keep=None)
    with tf.device('/gpu:'+str(cfg.GPU)):
     ckpt_file = os.path.join(cfg.OUTPUT_DIR, save_file_name)
     global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False) 
     learning_rate = tf.train.exponential_decay(
        initial_learning_rate, global_step, decay_steps,
        decay_rate, staircase, name='learning_rate')

     # Definition of evaluation-metric 
     correct_prediction = tf.equal(tf.argmax(network.logits,1), tf.argmax(network.labels,1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

     cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=network.labels, logits=network.logits))

     #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_function, global_step=global_step)
     optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(cost_function, global_step=global_step)

     ema = tf.train.ExponentialMovingAverage(decay=0.9999)
     averages_op = ema.apply(tf.trainable_variables())
     with tf.control_dependencies([optimizer]):
        train_op = tf.group(averages_op)
     sess = tf.InteractiveSession() #config=tf.ConfigProto(log_device_placement=True))
     #tf.global_variables_initializer().run()
     
     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
     sess.run(init_op)

     log_file = path_training_source_task+'log_files/log_file_'+dataset_name + dataset_spec + dataset_image_spec + '.txt'
     f = open(log_file, 'w')

     # Restoring weights from pre-trained model
     if weights_file is not None:
        f.write('\nRestoring weights from: ' + weights_file + '\n')
        #print('\nRestoring weights from: ' + weights_file + '\n\n')
        saver4restoring.restore(sess, weights_file)
    
    ##################
    # Code for training the model on the training-dataset
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    f.write('Start training...\n\n')
    for step in range(1, max_iter+1): 
       #print('---------------\niter: '+str(step)+':')
       # Declare training-batch
       ####images, labels = data.get()
       #images, labels = sess.run([images, annotations])
       batch_images, batch_labels = sess.run([images, annotations])
       #print(np.asarray(batch_labels).shape) 

       # Define forward-pass for training
       feed_dict = {network.images: batch_images, network.labels: batch_labels, network.keep_prob: 0.5}
       # Define forward-pass for testing
       feed_dict_test = {network.images: batch_images, network.labels: batch_labels, network.keep_prob: 1.0}
       # Display time, loss and accuracy
       if step%display_iter == 0 or step == 1:
          f = open(log_file, 'a')
          f.write('----------------------------------\n')
          #print("----------------------------------")
          f.write(str(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
          #print str(strftime("%Y-%m-%d %H:%M:%S", gmtime())),
          loss = sess.run(cost_function, feed_dict=feed_dict_test)
          acc = round(accuracy.eval(feed_dict=feed_dict_test),2)
          f.write(" Iteration: "+ str(step)+" - Loss: "+ str(loss) + " - Batch-Accuracy: " + str(acc) + "\n")
          #print("Iteration: "+ str(step)+" - Loss: "+ str(loss) + " - Batch-Accuracy: " + str(acc))
          f.close()

       #with tf.device('/gpu:'+str(cfg.GPU)):
       # # TRAINING
       sess.run(train_op, feed_dict=feed_dict)

       # SAVING 
       if step % save_iter == 0:
          f = open(log_file, 'a')
          f.write('Saving weights-file to: ' + str(ckpt_file) + '\n')
          #print('Saving weights-file to: ' + str(ckpt_file) + '\n')
          saver4saving.save(sess, ckpt_file, global_step=int(step))
          f.close()

    f.close()
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu X
    main()

