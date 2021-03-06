"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import cv2
from scipy import misc
import tensorflow.compat.v1 as tf
import numpy as np
import sys
import os
import copy
import imageio
import argparse
from src.align import facenet
import src.align.detect_face as detect_face
from six import moves
from PIL import Image

def main(args, img, Is_Restful):
    if Is_Restful:
        isface, images, cout_per_image, nrof_samples, img_aligned = load_and_align_data_rest(img, 160, 44, 1.0)
    else:
        isface, images, cout_per_image, nrof_samples, img_aligned = load_and_align_data(args.image_files, 160, 44, 1.0)

    if not isface:
        return False, None, None, None

    # images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            
            nrof_images = len(args.image_files)

            classifier_filename_exp = args.classifier_filename

            print('Testing classifier')
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
            predictions = model.predict_proba(emb)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            print(predictions)

            k=0
            for i in range(nrof_samples):
                for j in range(cout_per_image[i]):
                    print('%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
                    person = class_names[best_class_indices[k]]
                    confidence = best_class_probabilities[k] * 100
                    print(confidence)

                    if(confidence < 5):
                        confidence = None
                    k+=1


    result = {}
    result['detected']   = True
    result['person']     = person
    result['confidence'] = confidence

    return result


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    print(image_paths)
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ] # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.compat.v1.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = []
    count_per_image = []
    for i in moves.xrange(nrof_samples):
        img = imageio.imread(os.path.expanduser(image_paths[i]))

        # Check Channel Dimension of Image
        if len(img.shape) > 2 and img.shape[2] == 4:
            #convert the image from RGBA2RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        if len(bounding_boxes) == 0:
            return False, None, None, None, None

        j = 0
        count_per_image.append(1)
        det = np.squeeze(bounding_boxes[j,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned= np.array(Image.fromarray(cropped).resize(size=(image_size, image_size)))
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        images = np.stack(img_list)
        return True, images, count_per_image, nrof_samples, prewhitened

def load_and_align_data_rest(image_res, image_size, margin, gpu_memory_fraction):
    img = image_res
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ] # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.compat.v1.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    img_list = []
    count_per_image = []
    # Check Channel Dimension of Image
    if len(img.shape) > 2 and img.shape[2] == 4:
        #convert the image from RGBA2RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    if len(bounding_boxes) == 0:
        return False, None, None, None, None

    j = 0
    count_per_image.append(1)
    det = np.squeeze(bounding_boxes[j,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned= np.array(Image.fromarray(cropped).resize(size=(image_size, image_size)))
    prewhitened = facenet.prewhiten(aligned)
    img_list.append(prewhitened)
    images = np.stack(img_list)
    return True, images, count_per_image, 1, prewhitened

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',  help='Classifier model file name as a pickle (.pkl) file. ' +
        'For training this is the output and for classification this is an input.')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
