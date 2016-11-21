import os
import argparse

import numpy
import cv2
import tensorflow as tf

from nart import opr, aopr
from nart.model import VGG16


CONTENT_LAYERS = [('conv4_2',1.)]
STYLE_LAYERS = [('conv1_1',1.), ('conv2_1',1.5), ('conv3_1',2.), ('conv4_1',2.5), ('conv5_1',3.)]
STYLE_STRENTH = 500
IMAGE_NOISE_RATIO = 0.7

def main(args):
    net = VGG16(800, 600, args.weight_path)
    net.initialize()
    sess = tf.Session()

    img = cv2.imread(args.image_path)
    smg = cv2.imread(args.style_path)

    sess.run(tf.initialize_all_variables())
    sess.run(net['input'].assign(img))

    loss_content = -1
    for layer_name, layer_weight in CONTENT_LAYERS:
        ref = sess.run(net[layer_name])
        loss_content += layer_weight * aopr.na_content_loss(net[layer_name], ref)
    loss_style = 0
    for layer_name, layer_weight in STYLE_LAYERS:
        ref = sess.run(net[layer_name])
        loss_style += layer_weight * aopr.na_style_loss(net[layer_name], ref)
    loss = loss_content + loss_style * STYLE_STRENTH

    optimizer = tf.train.AdamOptimizer(2.0)
    fn_train = optimizer.minimize(loss)

    sess.run(tf.initialize_all_variables())
    sess.run(net['input'].assign(IMAGE_NOISE_RATIO * noise + (1. - IMAGE_NOISE_RATIO) * img))

    for i in range(100):
        sess.run(train)
        if i % 100 == 0:
            result_img = sess.run(net['input'])
            current_loss = sess.run(loss) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest='weight_path', required=True)
    parser.add_argument('-i', dest='image_path', required=True)
    parser.add_argument('-s', dest='style_path', required=True)

    main(parser.parse_args())

