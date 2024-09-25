# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:12:46 2021

@author: angelou
"""
import tensorflow as tf

class Conv(tf.keras.layers.Layer):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super(Conv, self).__init__()
        
        self.bn_acti = bn_acti
        self.padding = padding
        
        if isinstance(padding, str):
            self.conv = tf.keras.layers.Conv2D(filters=nOut,
                                           kernel_size=kSize,
                                           strides=stride,
                                           padding = padding,
                                           dilation_rate=dilation,
                                           groups=groups,
                                           use_bias=bias)
            self.pad_layer = None
        elif isinstance(padding, (int, tuple)):
            self.pad_layer = tf.keras.layers.ZeroPadding2D(padding = padding)
            self.conv = tf.keras.layers.Conv2D(filters=nOut,
                                           kernel_size=kSize,
                                           strides=stride,
                                           dilation_rate=dilation,
                                           groups=groups,
                                           use_bias=bias)
        
        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)
            
    def call(self, input):
        if self.pad_layer is None:
            output = self.conv(input)
        else:
            output = self.pad_layer(input)
            output = self.conv(output)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output  
    
class BNPReLU(tf.keras.layers.Layer):
    def __init__(self, nIn):
        super(BNPReLU, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-3)
        self.acti = tf.keras.layers.PReLU(shared_axes=[1, 2])

    def call(self, input):
        output = self.bn(input)
        output = self.acti(output)
        
        return output
