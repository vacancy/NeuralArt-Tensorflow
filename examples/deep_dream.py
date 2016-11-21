import os
import argparse

import cv2
import numpy as np
import tensorflow as tf

from nart import opr, aopr
from nart.model import VGG16
from nart.logconf import logger

LEARNING_RATE = 1.5
JITTER = 32

as_netin = lambda x: x[np.newaxis, :]

def make_step(sess, net, end):
    ''' iter only one step, providing end '''

    # random draw ox, oy
    ox, oy = np.random.randint(-JITTER, JITTER+1, 2)

    img = sess.run(net['input'])[0]
    img = np.roll(np.roll(img, ox, 1), oy, 0) # apply jitter shift

    # compute the gradient 
    # one shuold note that we are actually use L2 loss for an activation map to
    # to compute the gradient for the input
    sess.run(net['input'].assign(as_netin(img)))
    target = net[end]
    loss = 0.5 * tf.reduce_mean(tf.pow(target, 2))
    grad = tf.gradients(loss, [net['input']])[0]
    grad = sess.run(grad)[0]

    # apply gradient ascent, with normalized gradient
    img += LEARNING_RATE / np.abs(grad).mean() * grad
    img = np.clip(img, 0, 255)

    img = np.roll(np.roll(img, -ox, 1), -oy, 0) # unshift image
    sess.run(net['input'].assign(as_netin(img)))


def main(args):
    # read the image, and load the network
    img = cv2.imread(args.image_path)
    net = VGG16(args.weight_path, img.shape[0], img.shape[1])
    os.makedirs(args.output_path, exist_ok=True)

    # initialize the session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    sess.run(net['input'].assign(as_netin(img)))
    for i in range(0, args.nr_iters+1):
        if i != 0:
            make_step(sess, net, end=args.end)

        # save the result image every ``args.save_step'' iterations
        if i % args.save_step == 0:
            current_img = sess.run(net['input'])[0]

            output_path = os.path.join(args.output_path, 'epoch_{:04d}.png'.format(i))
            cv2.imwrite(output_path, current_img)
            logger.info('epoch {}: image written to {}'.format(i, output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest='weight_path', required=True, help='weight path')
    parser.add_argument('-i', dest='image_path', required=True, help='input image path')
    parser.add_argument('-o', dest='output_path', required=True, help='output directory')
    parser.add_argument('-e', '--end', dest='end', default='conv5_3', help='end')
    parser.add_argument('--iter', dest='nr_iters', type=int, default=100, help='number of iterations')
    parser.add_argument('--save-step', dest='save_step', type=int, default=5, help='save step (in iteration)')

    main(parser.parse_args())

