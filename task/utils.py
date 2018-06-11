import tensorflow as tf
import numpy as np
import os
from scipy import misc
from os.path import isfile, join


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Author: Jin Yamanaka
Github: https://github.com/jiny2001/dcscn-image-super-resolution
Create Augmented training images
Put your images under data/[your dataset name]/ and specify [your dataset name] for --dataset.
--augment_level 2-8: will generate flipped / rotated images
"""


def data_augmentation(augment_level=8):
    print("Building x%d augmented data." % augment_level)

    training_filename = get_files_in_directory("data/preinput_images")
    target_dir = "data/input_images/"

    for file_path in training_filename:
        org_image = misc.imread(file_path)
        filename = os.path.basename(file_path)
        filename, extension = os.path.splitext(filename)

        new_filename = target_dir + filename
        misc.imsave(new_filename + extension, org_image)

        if augment_level >= 2:
            ud_image = np.flipud(org_image)
            misc.imsave(new_filename + "_v" + extension, ud_image)
        if augment_level >= 3:
            lr_image = np.fliplr(org_image)
            misc.imsave(new_filename + "_h" + extension, lr_image)
        if augment_level >= 4:
            lr_image = np.fliplr(org_image)
            lrud_image = np.flipud(lr_image)
            misc.imsave(new_filename + "_hv" + extension, lrud_image)
        if augment_level >= 5:
            rotated_image1 = np.rot90(org_image)
            misc.imsave(new_filename + "_r1" + extension, rotated_image1)
        if augment_level >= 6:
            rotated_image2 = np.rot90(org_image, -1)
            misc.imsave(new_filename + "_r2" + extension, rotated_image2)
        if augment_level >= 7:
            rotated_image1 = np.rot90(org_image)
            ud_image = np.flipud(rotated_image1)
            misc.imsave(new_filename + "_r1_v" + extension, ud_image)
        if augment_level >= 8:
            rotated_image2 = np.rot90(org_image, -1)
            ud_image = np.flipud(rotated_image2)
            misc.imsave(new_filename + "_r2_v" + extension, ud_image)


def get_files_in_directory(path):
    if not path.endswith('/'):
        path = path + "/"
    file_list = [path + f for f in os.listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]
    return file_list


if __name__ == '__main__':
    tf.app.run()
