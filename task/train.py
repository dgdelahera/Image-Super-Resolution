from __future__ import print_function, division

from task import models
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('scale_factor', 2,
                            """Scale factor.""")
tf.app.flags.DEFINE_integer('num_epochs', 250,
                            """Size of batches.""")
tf.app.flags.DEFINE_string('img_dir', 'data/input_images/',
                           """Directory where to find images input """)
tf.app.flags.DEFINE_string('val_dir', 'data/val_images/',
                           """Directory where to find validation images """)
tf.app.flags.DEFINE_string('save_dir', 'data/save',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('mode', 'patch',
                           """Mode of operation. Choices are "fast" or "patch.""")
tf.app.flags.DEFINE_integer('patch_size', 8,
                            """Size of patches.""")
tf.app.flags.DEFINE_string('suffix', 'str',
                           """Suffix of saved image.""")


def train():
    #sr = models.ImageSuperResolutionModel(FLAGS.scale_factor)
    #sr.create_model()
    # Check small_images transform_image true_upscale
    #sr.fit(nb_epochs=FLAGS.num_epochs)

    dsr = models.DenoisingAutoEncoderSR(scale_factor=2)
    dsr.create_model(width=64, height=64)
    dsr.fit(nb_epochs=250)


def main(argv=None):
    train()
    print("as")

if __name__ == "__main__":
    tf.app.run()