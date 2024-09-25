import tensorflow as tf
from .conv_layer import Conv

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, in_channels, mode='hw'):
        super(SelfAttention, self).__init__()

        self.mode = mode

        self.query_conv = Conv(nIn=in_channels, nOut=in_channels // 8, kSize=(1, 1), stride=1, padding='VALID')
        self.key_conv = Conv(nIn=in_channels, nOut=in_channels // 8, kSize=(1, 1), stride=1, padding='VALID')
        self.value_conv = Conv(nIn=in_channels, nOut=in_channels, kSize=(1, 1), stride=1, padding='VALID')

        self.gamma = tf.Variable(tf.zeros(1), trainable=True)
        self.sigmoid = tf.keras.activations.sigmoid

    def call(self, x):
        batch_size, height, width, channel = x.get_shape().as_list()
        
        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width
        
    
        view = (-1, axis)

        projected_query_0 = self.query_conv(x)
        projected_query_1 = tf.keras.layers.Reshape(view)(projected_query_0)
        projected_query = tf.transpose(projected_query_1, perm=[0, 2, 1])
        projected_key = tf.keras.layers.Reshape(view)(self.key_conv(x))

        attention_map = tf.matmul(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        attention = tf.transpose(attention, perm=[0, 2, 1])
        projected_value = self.value_conv(x)
        projected_value = tf.keras.layers.Reshape(view)(projected_value)

        out = tf.matmul(projected_value, tf.transpose(attention, perm=[0, 2, 1]))
        
        out = tf.keras.layers.Reshape((height, width, channel))(out)

        out = self.gamma * out + x
        return out
