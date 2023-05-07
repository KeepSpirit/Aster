from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


def shuffle_unit(inputs, out_channels, strides=1):
    # Define the number of input channels
    in_channels = inputs.shape[-1]

    # Define the bottleneck channels
    bottleneck_channels = out_channels // 2

    # Split the input channels into two equal parts
    x1, x2 = inputs[:, :, :, :in_channels // 2], inputs[:, :, :, in_channels // 2:]

    # Apply a pointwise convolution to the second part of the input
    x2 = Conv2D(filters=bottleneck_channels, kernel_size=(1, 1), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    # Apply a depthwise convolution to the second part of the input
    x2 = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    # Apply a pointwise convolution to the second part of the input
    x2 = Conv2D(filters=bottleneck_channels, kernel_size=(1, 1), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    # Modify the shape of x1 to match the shape of x2 along the concatenation axis
    if strides == 1 and in_channels == out_channels:
        x1_downsample = x1
    else:
        x1_downsample = Conv2D(filters=bottleneck_channels, kernel_size=(1, 1), padding='same')(x1)
        x1_downsample = BatchNormalization()(x1_downsample)
        x1_downsample = Activation('relu')(x1_downsample)
        x1_downsample = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(x1_downsample)
        x1_downsample = BatchNormalization()(x1_downsample)
        x1_downsample = Activation('relu')(x1_downsample)
        x1_downsample = Conv2D(filters=bottleneck_channels, kernel_size=(1, 1), padding='same')(x1_downsample)
        x1_downsample = BatchNormalization()(x1_downsample)
        x1_downsample = Activation('relu')(x1_downsample)

    # Concatenate the two parts of the input
    x = Concatenate()([x1_downsample, x2])
    return x


def shuffle_stage(inputs, out_channels, num_blocks, strides=2):
    # Apply the first shuffle unit with strided convolution
    x = shuffle_unit(inputs, out_channels, strides=strides)

    # Apply the remaining shuffle units with stride=1
    for _ in range(num_blocks - 1):
        x = shuffle_unit(x, out_channels)

    return x


def ShuffleNetV2(input_shape, num_classes):
    # Define the input tensor
    inputs = Input(shape=input_shape)

    # Apply the stem convolution
    x = Conv2D(filters=24, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Apply the shuffle stages
    x = shuffle_stage(x, out_channels=116, num_blocks=2, strides=2)
    x = shuffle_stage(x, out_channels=232, num_blocks=3, strides=2)
    x = shuffle_stage(x, out_channels=464, num_blocks=2, strides=2)

    # Apply the final convolution and pooling
    x = Conv2D(filters=1024, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # Apply the final dense layer
    outputs = Dense(units=num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model
