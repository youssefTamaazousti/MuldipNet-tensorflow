import numpy as np
import tensorflow as tf
import datetime
from PIL import Image
from glob import glob
from time import gmtime, strftime
import os, sys, inspect
# To import modules that are in the parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
# import config file
import utils.config_training as cfg

def main():
    ##################
    #   Parameters   #
    ##################
    # Declare argument-parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='alexnet', help='Name of the network-architecture to use')
    parser.add_argument('--batch_size', type=int, default=256, help='Amount of data used in mini-batch')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device on which are declared the variables')
    args = parser.parse_args()
    # Get the network-architecture from the arguments  
    if args.architecture=="alexnet":
       from utils.architecture_alexnet_forTraining import NET
    elif args.architecture=="darknet":
       from utils.architecture_darknet_forTraining import NET
    print("architecture = "+str(args.architecture))
    
    dataset = cfg.DATASET
    source_problem = cfg.SOURCE_PROBLEM
    images_spec = cfg.IMAGES_SPEC
    display_iter = cfg.DISPLAY_ITER
    max_iter = cfg.MAX_ITER
    save_iter = cfg.SAVE_ITER
    weights_file = cfg.WEIGHTS_FILE
    image_resize = cfg.IMAGE_RESIZE
    image_size = cfg.IMAGE_SIZE
    initial_learning_rate = cfg.LEARNING_RATE
    decay_steps = cfg.DECAY_STEPS
    decay_rate = cfg.DECAY_RATE
    staircase = cfg.STAIRCASE
    momentum = cfg.MOMENTUM
    number_categories = cfg.NUMBER_CATEGORIES

    # Infer paths and output file names
    path_data = '../data/'+dataset+'/'
    output_models_dir = path_data + 'models/' + 'models.'+dataset+'.'+source_problem+'/'
    # Check if directory exists and create one if it doesn't:
    if not os.path.isdir(output_models_dir):
       os.makedirs(output_models_dir)
    output_model = output_models_dir + model.'+dataset+'_'+source_problem+'_'+images_spec+'.config.number_categories='+str(number_categories)+'_imgSize='+str(IMAGE_SIZE)+'_imgSpec='+str(IMAGES_SPEC)+'.ckpt'
    mean_values_filename = path_data + 'mean_values/mean_values.'+ dataset_name + dataset_spec  +'.txt'
    tfrecords_filename = path_data +'tfrecord_files/tfrecord_files'+dataset_spec +'_'+dataset_image_spec +'.train.bin'
    
    # Reading mean-values
    with open(mean_values_filename, 'r') as mean_values_file:
       mean_values = [x.strip() for x in mean_values_file.readlines()]  
    R_mean = float(mean_values[0])
    G_mean = float(mean_values[1])
    B_mean = float(mean_values[2])

    ###########################
    # READING TF-RECORD files #
    ###########################
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

    # Cast label data into int64
    label = tf.cast(features['annotation_raw'], tf.int64)

    # Pre-process images (resize, random crop and random flip)
    image = tf.image.resize_images (image, [image_resize, image_resize])
    image = tf.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    mean = tf.constant([R_mean, G_mean, B_mean], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 1, 3])
    image = image - mean
    
    label_to_feed = tf.one_hot(label, number_categories, on_value=1, off_value=0)
    label_to_feed = tf.cast(label_to_feed, tf.float32)

    # Create batches by randomly shuffling tensors
    images, annotations = tf.train.batch([image, label_to_feed], batch_size=args.batch_size, capacity=30, num_threads=3) # you can set the num_threads variable to a higher value for speeding-up the learning process
    
    # Working on gpu device (if you want to work on CPU, remove the "with tf.device('/gpu:'+str(args.gpu)):" lines)
    with tf.device('/gpu:'+str(args.gpu)):
     # Declare network
     network = NET()

     #############################
     # Declare solver-parameters #
     #############################
     # Set of variables that we will restore 
     variable_to_restore = tf.global_variables() 
     # Declare the saver for restoring the set of variables in variable_to_restore
     saver4restoring = tf.train.Saver(variable_to_restore, max_to_keep=None)    
    # Declare the saver for saving all trainable variables 
    saver4saving = tf.train.Saver(max_to_keep=None)
    
    with tf.device('/gpu:'+str(args.gpu)):
     global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False) 
     learning_rate = tf.train.exponential_decay(
        initial_learning_rate, global_step, decay_steps,
        decay_rate, staircase, name='learning_rate')

     # Definition of evaluation-metric (accuracy)
     correct_prediction = tf.equal(tf.argmax(network.logits,1), tf.argmax(network.labels,1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     
     # Definition of the cost-function (softmax cross-entropy)
     cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=network.labels, logits=network.logits))

     # Definition of the optimizer (SGD with momentum)
     optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(cost_function, global_step=global_step)

     ema = tf.train.ExponentialMovingAverage(decay=0.9999)
     averages_op = ema.apply(tf.trainable_variables())
     with tf.control_dependencies([optimizer]):
        train_op = tf.group(averages_op)
     sess = tf.InteractiveSession() 
     
     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
     sess.run(init_op)

     # Restoring weights from pre-trained model
     if weights_file is not None:
        print('\nRestoring weights from: ' + weights_file + '\n\n')
        saver4restoring.restore(sess, weights_file)
    
    ##################
    #    Training    #
    ##################
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print('Start training...\n\n')
    for step in range(1, max_iter+1): 
       print('---------------\niter: '+str(step)+':')
       # Declare training-batch
       batch_images, batch_labels = sess.run([images, annotations])

       # Define forward-pass for training
       feed_dict = {network.images: batch_images, network.labels: batch_labels, network.keep_prob: 0.5}
       # Define forward-pass for testing
       feed_dict_test = {network.images: batch_images, network.labels: batch_labels, network.keep_prob: 1.0}
       # Display time, loss and accuracy (on the next training-batch)
       if step%display_iter == 0 or step == 1:
          print("----------------------------------")
          print str(strftime("%Y-%m-%d %H:%M:%S", gmtime())),
          loss = sess.run(cost_function, feed_dict=feed_dict_test)
          acc = round(accuracy.eval(feed_dict=feed_dict_test),2)
          print("Iteration: "+ str(step)+" - Loss: "+ str(loss) + " - Batch-Accuracy: " + str(acc))
       # Perform optimization
       sess.run(train_op, feed_dict=feed_dict)

       # Save the model 
       if step % save_iter == 0:
          f = open(log_file, 'a')
          print('Saving weights-file to: ' + str(output_model) + '\n')
          saver4saving.save(sess, output_model, global_step=int(step))

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()

