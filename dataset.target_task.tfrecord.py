from PIL import Image
import numpy as np
import tensorflow as tf
import random

########################
#      Parameters      #
########################
for dataset in ['voc07', 'voc12', 'ca101', 'ca256', 'mit67', 'stCAR', 'cub200', 'nwOB', 'stACT', 'fl102']: # set the directory name of your target-dataset here
 for phase in ['train', 'test']:

  print ('Generating tfrecord-file for '+phase+' phase of '+dataset+' dataset')
  path_to_data = '/data/target_tasks/'+dataset+'/'
  txtname = path_to_data + 'lists/' + dataset + '.' + phase + '.lst'
  tfrecords_filename = path_to_data + 'tfrecord_files/tfrecord_files' + dataset + '.'  + phase+'.bin'

  ########################
  # ImageCoder class
  ########################
  class ImageCoder(object):  
    def __init__(self):
      # Create a single Session to run all image coding calls.
      self._sess = tf.Session()

      # Initializes function that converts PNG to JPEG data.
      self._png_data = tf.placeholder(dtype=tf.string)
      image = tf.image.decode_png(self._png_data, channels=3)
      self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
      # Initializes function that converts CMYK JPEG data to RGB JPEG data.
      self._cmyk_data = tf.placeholder(dtype=tf.string)
      image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
      self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)
      # Initializes function that decodes RGB JPEG data.
      self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
      self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
      return self._sess.run(self._png_to_jpeg,
                            feed_dict={self._png_data: image_data})
    def cmyk_to_rgb(self, image_data):
      return self._sess.run(self._cmyk_to_rgb,
                            feed_dict={self._cmyk_data: image_data})
    def decode_jpeg(self, image_data):
      image = self._sess.run(self._decode_jpeg,
                             feed_dict={self._decode_jpeg_data: image_data})
      assert len(image.shape) == 3
      assert image.shape[2] == 3
      return image
  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()
  
  with open(txtname, 'r') as f:
     imagepath_labels = [x.strip() for x in f.readlines()]

  img_paths = []
  for annot_index in imagepath_labels:
     img_paths.append(annot_index)

  def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _int64_feature(value):
    # Wrapper for inserting int64 features into Example proto.
    if not isinstance(value, list):
      value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _float_feature(value):
      if not isinstance(value, list):
         value = [value]
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def one_hot(index, depth):
      vector = np.zeros((depth), dtype=np.int)
      vector[index] = 1
      return vector

  writer = tf.python_io.TFRecordWriter(tfrecords_filename)
  print('Saving tfrecord file for all the dataset...')
  for i in range(len(img_paths)): 
      if i%1000 == 0:
         print(str(i)+'/'+str(len(img_paths)))
      # Read the image file.
      with tf.gfile.FastGFile(img_paths[i], 'r') as f:
         img_raw = f.read()
      # Decode the RGB JPEG.
      image = coder.decode_jpeg(img_raw)
      height = image.shape[0]
      width = image.shape[1]  
    
      example = tf.train.Example(features=tf.train.Features(feature={
          'height': _int64_feature(height),
          'width': _int64_feature(width),
          'image_raw': _bytes_feature(img_raw)}))
      writer.write(example.SerializeToString())
 
  writer.close()
