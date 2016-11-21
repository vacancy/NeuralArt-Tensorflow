import os
import argparse

import cv2
import numpy as np
import tensorflow as tf

from nart import opr, aopr
from nart.model import VGG16
from nart.logconf import logger

CONTENT_LAYERS = [('conv4_2',1.)]
STYLE_LAYERS = [('conv1_1',1.), ('conv2_1',1.5), ('conv3_1',2.), ('conv4_1',2.5), ('conv5_1',3.)]
STYLE_STRENTH = 500
IMAGE_NOISE_RATIO = 0.7


def main(args):
    as_netin = lambda x: x[np.newaxis, :]

    # read images
    img = cv2.imread(args.image_path)
    smg = cv2.imread(args.style_path)
    smg = cv2.resize(smg, (img.shape[1], img.shape[0]))

    # building the net providing the path to the weights
    net = VGG16(args.weight_path, img.shape[0], img.shape[1])
    os.makedirs(args.output_path, exist_ok=True)

    # initialize the session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # compute the content loss
    loss_content = -1
    sess.run(net['input'].assign(as_netin(img)))
    for layer_name, layer_weight in CONTENT_LAYERS:
        ref = sess.run(net[layer_name])
        loss_content += layer_weight * aopr.na_content_loss(net[layer_name], ref)

    # compute the style loss
    loss_style = 0
    sess.run(net['input'].assign(as_netin(smg)))
    for layer_name, layer_weight in STYLE_LAYERS:
        ref = sess.run(net[layer_name])
        loss_style += layer_weight * aopr.na_style_loss(net[layer_name], ref)

    # sum up the loss
    loss = loss_content / STYLE_STRENTH + loss_style

    # use Adam Optimizer
    optimizer = tf.train.AdamOptimizer(2.0)
    fn_train = optimizer.minimize(loss)

    # initialize the variables again to prepare the training
    sess.run(tf.initialize_all_variables())
  
    # add noise to the input, and assign it to the network's input
    noise = np.random.uniform(-20, 20, img.shape).astype('float32')
    noise_img = IMAGE_NOISE_RATIO * noise + (1. - IMAGE_NOISE_RATIO) * img
    sess.run(net['input'].assign(as_netin(noise_img)))

    for i in range(0, args.nr_iters+1):
        if i != 0:
            sess.run(fn_train)

        # save the result image and output the current loss every ``args.save_step'' iterations
        if i % args.save_step == 0:
            current_img = sess.run(net['input'])[0]
            current_loss = sess.run(loss) 
            
            output_path = os.path.join(args.output_path, 'epoch_{:04d}.png'.format(i))
            cv2.imwrite(output_path, current_img)
            logger.info('epoch {}: loss={}, image written to {}'.format(i, current_loss, output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest='weight_path', required=True, help='weight path')
    parser.add_argument('-i', dest='image_path', required=True, help='input image path')
    parser.add_argument('-s', dest='style_path', required=True, help='style image path')
    parser.add_argument('-o', dest='output_path', required=True, help='output directory')
    parser.add_argument('--iter', dest='nr_iters', type=int, default=1000, help='number of iterations')
    parser.add_argument('--save-step', dest='save_step', type=int, default=50, help='save step (in iteration)')

    main(parser.parse_args())

