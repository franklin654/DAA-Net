# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:17:13 2021

@author: angelou
"""
import tensorflow as tf
from .conv_layer import Conv
from .self_attention import SelfAttention
import math

class AA_kernel(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(nIn=in_channel, nOut=out_channel, kSize=(1, 1), stride=1, padding='VALID')
        self.conv1 = Conv(nIn=in_channel, nOut=out_channel, kSize=(3, 3), stride=1, padding='SAME')
        self.Hattn = SelfAttention(out_channel, mode='h')
        self.Wattn = SelfAttention(out_channel, mode='w')

    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx
