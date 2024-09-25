# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:20:30 2021

@author: angelou
"""
import tensorflow as tf
from .conv_layer import Conv, BNPReLU
import math
class aggregation(tf.keras.layers.Layer):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = tf.keras.layers.ReLU()

        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_upsample1 = Conv(32, 32, 3, 1, padding='same')
        self.conv_upsample2 = Conv(32, 32, 3, 1, padding='same')
        self.conv_upsample3 = Conv(32, 32, 3, 1, padding='same')
        self.conv_upsample4 = Conv(32, 32, 3, 1, padding='same')
        self.conv_upsample5 = Conv(64, 64, 3, 1, padding='same')  # Adjust channels based on concatenation

        self.conv_concat2 = Conv(64, 64, 3, 1, padding='same')
        self.conv_concat3 = Conv(96, 96, 3, 1, padding='same')
        self.conv4 = Conv(96, 96, 3, 1, padding='same')
        self.conv5 = tf.keras.layers.Conv2D(96, 1, 1)  # Final output channel

    def call(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = tf.concat((x2_1, self.conv_upsample4(self.upsample(x1_1))), axis=-1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = tf.concat((x3_1, self.conv_upsample5(self.upsample(x2_2))), axis=-1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x
