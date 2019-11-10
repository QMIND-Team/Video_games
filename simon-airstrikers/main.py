import tensorflow as tf
from src.inference import train, eval_play

#The first is the parameter name, the second is the default value, and the third is the parameter description.
tf.compat.v1.flags.DEFINE_integer('train', 1, "whether do training(1) or testing(0)")

FLAGS = tf.compat.v1.flags.FLAGS

def main(_):
    with tf.compat.v1.Session() as sess:
        if FLAGS.train == 1: 
            train(sess)
        elif FLAGS.train == 0: 
            eval_play(sess)
        else:
            raise ValueError('train parameter should be 0 or 1')

if __name__ == '__main__':
    tf.compat.v1.app.run()