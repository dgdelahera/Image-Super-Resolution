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


def test():
    """
    Train Super Resolution
    """
    action = input("\n"
                   "Type the number of action you would like to do:\n"
                   "\t[1] Train the model.\n"
                   "\t[2] Eval the model.\n"
                   "\t[3] Upscale an image.\n\n")

    if action not in ['1', '2', '3']:
        print("Invalid action.")
        return 0
    if action == "1":
        mode = input("\n"
                     "Would you like to restore the previous weights?:\n"
                     "\t[1] Yes.\n"
                     "\t[2] No.\n\n")
        if mode == "1":
            sr = models.ImageSuperResolutionModel(FLAGS.scale_factor)
            sr.create_model(height=64, width=64, load_weights=True)
            sr.fit(nb_epochs=FLAGS.num_epochs)
        if mode == "2":
            sr = models.ImageSuperResolutionModel(FLAGS.scale_factor)
            sr.create_model(height=64, width=64, load_weights=False)
            sr.fit(nb_epochs=FLAGS.num_epochs)

    if action == "2":
        sr = models.ImageSuperResolutionModel(FLAGS.scale_factor)
        sr.evaluate(FLAGS.val_dir)

    if action == "3":
        # with tf.device('/CPU:0'):
        image = input("\n"
                     "Type the image to upscale:\n\n")
        model = models.ImageSuperResolutionModel(FLAGS.scale_factor)
        model.upscale("data/examples/" + image, save_intermediate=True, mode=FLAGS.mode, patch_size=FLAGS.patch_size,
                      suffix=FLAGS.suffix)



def main(argv=None):
    test()


if __name__ == "__main__":
    tf.app.run()
