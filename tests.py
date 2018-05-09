from __future__ import print_function, division

import models
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('scale_factor', 2,
                            """Scale factor.""")
tf.app.flags.DEFINE_integer('num_epochs', 250,
                            """Size of batches.""")
tf.app.flags.DEFINE_string('img_dir', 'data/input_images',
                           """Directory where to find images input """)
tf.app.flags.DEFINE_string('save_dir', 'data/pixel',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('mode', 'patch',
                           """Mode of operation. Choices are "fast" or "patch.""")
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Size of batches.""")
tf.app.flags.DEFINE_string('suffix', 'str',
                           """Suffix of saved image.""")

def test():

    """
    Train Super Resolution
    """

    sr = models.ImageSuperResolutionModel(FLAGS.scale_factor)
    sr.create_model(load_weights=False)
    sr.fit(nb_epochs=FLAGS.num_epochs)

    # """
    # Evaluate Super Resolution on Set5/14
    # """
    #
    # sr = models.ImageSuperResolutionModel(scale)
    # sr.evaluate(val_path)
    #
    #
    # """
    # Compare output images of sr, esr, dsr and ddsr models
    # """
    # # Main:::
    #    with tf.device('/CPU:0'):
    #   model = models.ImageSuperResolutionModel(FLAGS.scale_factor)
    #    model.upscale(FLAGS.img_dir, save_intermediate=FLAGS.save_dir, mode=FLAGS.mode, patch_size=FLAGS.batch_size,
    #                 suffix=FLAGS.suffix)

    # sr = models.ImageSuperResolutionModel(scale)
    # sr.upscale(path, save_intermediate=False, suffix="sr")
    #
    # gansr = models.GANImageSuperResolutionModel(scale)
    # gansr.upscale(path, save_intermediate=False, suffix='gansr')


def main(argv=None):
    test()


if __name__ == "__main__":
    tf.app.run()



