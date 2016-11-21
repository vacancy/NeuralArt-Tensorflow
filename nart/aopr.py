# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# 
# This file is part of LibNeuralArt 

''' artistic oprs, used for creating arts '''

def na_content_loss(inp, ref):
    n = ref.shape[1] * ref.shape[2]
    c = ref.shape[3]

    loss = (1. / (2. * n ** 0.5 * c ** 0.5)) * tf.reduce_sum(tf.pow((inp - ref), 2))
    return loss


def na_style_loss(inp, ref):
    n = ref.shape[1] * ref.shape[2]
    c = ref.shape[3]

    ref = ref.reshape(n, c)
    inp = tf.reshape(inp, (n, c))

    gref = np.dot(ref.T, ref)
    ginp = tf.matmul(tf.transpose(inp), inp)

    loss = (1. / (4. * n ** 2 * c ** 0.5)) * tf.reduce_sum(tf.pow(ginp - gref, 2))
    return loss

