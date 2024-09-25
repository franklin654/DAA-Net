import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D
from keras.layers import add
from keras.models import Model

from CustomLayers.ConvBlock2D import conv_block_2D
from .axial_atten import AA_kernel
from .context_module import CFPModule
from .partial_decoder import aggregation

kernel_initializer = 'he_uniform'
interpolation = "nearest"


def create_model(img_height, img_width, input_chanels, out_classes, starting_filters, blocks=['duckv2' for i in range(5)]):
    input_layer = tf.keras.layers.Input((img_height, img_width, input_chanels))

    print('Starting DUCK-Net')

    p1 = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(input_layer)
    p2 = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(p1)
    p3 = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(p2)
    p4 = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(p3)
    p5 = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(p4)

    # 1st duck_block
    first_block = blocks[0]
    t0 = conv_block_2D(input_layer, starting_filters, first_block, repeat=1)

    l1i = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(t0)
    s1 = add([l1i, p1])
    # 2nd duck_block
    second_block = blocks[1]
    t1 = conv_block_2D(s1, starting_filters * 2, second_block, repeat=1)

    l2i = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(t1)
    s2 = add([l2i, p2])
    # 3rd duck_block
    t2 = conv_block_2D(s2, starting_filters * 4, blocks[2], repeat=1)

    l3i = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(t2)
    s3 = add([l3i, p3])
    # 4th duck_block
    # fourth_block = hp.Choice("fourth_block", ['seperated', 'duckv2', 'midscope', 'widescope', 'resnet', 'conv'], defualt='duckv2')
    t3 = conv_block_2D(s3, starting_filters * 8, blocks[3], repeat=1)

    l4i = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(t3)
    s4 = add([l4i, p4])
    # 5th duck_block
    # fifth_block = hp.Choice("fifth_block",  ['seperated', 'duckv2', 'midscope', 'widescope', 'resnet', 'conv'], default='duckv2')
    t4 = conv_block_2D(s4, starting_filters * 16, blocks[4], repeat=1)

    l5i = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(t4)
    s5 = add([l5i, p5])
    t51 = conv_block_2D(s5, starting_filters * 32, 'resnet', repeat=2)
    t53 = conv_block_2D(t51, starting_filters * 16, 'resnet', repeat=2)

    # attention one
    decoder_one = UpSampling2D((2, 2), interpolation=interpolation)(t53)
    cfp_out_1 = CFPModule(starting_filters*16, d=8)(t4)
    decoder_one_ra = -1*(tf.keras.activations.sigmoid(decoder_one)) + 1
    att_1 = AA_kernel(starting_filters*16, starting_filters*16)(cfp_out_1)
    att_1_out = tf.keras.layers.Multiply()([decoder_one_ra, att_1])
    q4_in = tf.keras.layers.Add()([att_1_out, decoder_one])
    q4 = conv_block_2D(q4_in, starting_filters * 8, 'duckv2', repeat=1)

    # deep supervision one
    lateral_map_1 = Conv2D(1, 1, activation='relu')(q4)
    lateral_map_1 = UpSampling2D((4, 4), interpolation='bilinear')(lateral_map_1)
    lateral_map_1 = tf.keras.layers.Activation('sigmoid', name="lateral_map_1")(lateral_map_1)
    

    # attention two
    decoder_two = UpSampling2D((2, 2), interpolation=interpolation)(q4)
    decoder_two_ra = -1*(tf.keras.activations.sigmoid(decoder_two)) + 1
    cfp_out_2 = CFPModule(starting_filters * 8, d=8)(t3)
    # c3 = add([l4o, cfp_out_2])
    att_2 = AA_kernel(starting_filters*8, starting_filters*8)(cfp_out_2)
    att_2_out = tf.keras.layers.Multiply()([decoder_two_ra, att_2])
    q3 = tf.keras.layers.Add()([att_2_out, decoder_two])
    q3 = conv_block_2D(q3, starting_filters * 4, 'duckv2', repeat=1)

    # deep supervision one
    lateral_map_2 = Conv2D(1, 1, activation='relu')(q3)
    lateral_map_2 = UpSampling2D((4, 4), interpolation='bilinear')(lateral_map_2)
    lateral_map_2 = tf.keras.layers.Activation('sigmoid', name="lateral_map_2")(lateral_map_2)

    # attention three
    decoder_three = UpSampling2D((2, 2), interpolation=interpolation)(q3)
    decoder_three_ra = -1*(tf.keras.activations.sigmoid(decoder_three)) + 1
    cfp_out_3 = CFPModule(starting_filters*4, d=8)(t2)
    # c2 = add([l3o, cfp_out_3])
    att_3 = AA_kernel(starting_filters*4, starting_filters*4)(cfp_out_3)
    att_3_out = tf.keras.layers.Multiply()([decoder_three_ra, att_3])
    q2_in = tf.keras.layers.Add()([att_3_out, decoder_three])
    q2 = conv_block_2D(q2_in, starting_filters * 2, 'duckv2', repeat=1)

    # deep supervision three
    lateral_map_3 = Conv2D(1, 1, activation='relu')(q2)
    lateral_map_3 = UpSampling2D((4, 4), interpolation='bilinear')(lateral_map_3)
    lateral_map_3 = tf.keras.layers.Activation('sigmoid', name="lateral_map_3")(lateral_map_3)

    # attention four
    decoder_four = UpSampling2D((2, 2), interpolation=interpolation)(q2)
    decoder_four_ra = -1*(tf.keras.activations.sigmoid(decoder_four)) + 1
    cfp_out_4 = CFPModule(starting_filters*2, d=8)(t1)
    # c1 = add([l2o, cfp_out_4])
    att_4 = AA_kernel(starting_filters*2, starting_filters*2)(cfp_out_4)
    att_4_out = tf.keras.layers.Multiply()([decoder_four_ra, att_4])
    q1 = tf.keras.layers.Add()([att_4_out, decoder_four])
    q1 = conv_block_2D(q1, starting_filters, 'duckv2', repeat=1)
    

    l1o = UpSampling2D((2, 2), interpolation=interpolation)(q1)
    c0 = add([l1o, t0])
    z1 = conv_block_2D(c0, starting_filters, 'duckv2', repeat=1)

    output = Conv2D(out_classes, (1, 1), activation='sigmoid', name='output')(z1)

    model = Model(inputs=input_layer, outputs=[output, lateral_map_3])
    """, lateral_map_2, lateral_map_1]"""

    return model
