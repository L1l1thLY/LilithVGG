import vgg16
import tensorflow as tf
import image_tools
import score_tools
import numpy as np

class Lilith_vgg(vgg16.Vgg16):
    def __init__(self, true_sex_false_age=True, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            print("Vgg16: Do not find npy file.")
            # TODO: Find a default path properly.

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

        print("Vgg16: npy file loaded")
        self.sex_or_age = true_sex_false_age
        self.var_dict = {}

    def build(self, rgb_image):
        if self.sex_or_age is True:
            self.build_sex_vgg16(rgb_image)
        else:
            self.build_age_vgg16(rgb_image)

    def build_sex_vgg16(self, rgb_image, name="vgg_sex"):
        pass

    def build_age_vgg16(selfs, rgb_image, name="vgg_age"):
        pass