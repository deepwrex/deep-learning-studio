import tensorflow as tf


MODEL_DICT = {
    'VGG16': [
        tf.keras.applications.vgg16.VGG16,
        tf.keras.applications.vgg16.preprocess_input,
        tf.keras.applications.vgg16.decode_predictions,
        'block5_conv3', (224, 224),
        ['block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']
    ],
    'VGG19': [
        tf.keras.applications.vgg19.VGG19,
        tf.keras.applications.vgg19.preprocess_input,
        tf.keras.applications.vgg19.decode_predictions,
        'block5_conv4', (224, 224),
        ['block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']
    ],
    'ResNet50': [
        tf.keras.applications.resnet50.ResNet50,
        tf.keras.applications.resnet50.preprocess_input,
        tf.keras.applications.resnet50.decode_predictions,
        'conv5_block3_out', (224, 224),
        ['avg_pool', 'predictions']
    ],
    'ResNet101': [
        tf.keras.applications.resnet.ResNet101,
        tf.keras.applications.resnet.preprocess_input,
        tf.keras.applications.resnet.decode_predictions,
        'conv5_block3_out', (224, 224),
        ['avg_pool', 'predictions']
    ],
    'ResNet152': [
        tf.keras.applications.resnet.ResNet152,
        tf.keras.applications.resnet.preprocess_input,
        tf.keras.applications.resnet.decode_predictions,
        'conv5_block3_out', (224, 224),
        ['avg_pool', 'predictions']
    ],
    'MobileNet-V1': [
        tf.keras.applications.mobilenet.MobileNet,
        tf.keras.applications.mobilenet.preprocess_input,
        tf.keras.applications.mobilenet.decode_predictions,
        'conv_pw_13_relu', (224, 224),
        [
            'global_average_pooling2d', 'reshape_1', 'dropout',
            'conv_preds', 'reshape_2', 'predictions'
        ]
    ],
    'MobileNet-V2': [
        tf.keras.applications.mobilenet_v2.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
        tf.keras.applications.mobilenet_v2.decode_predictions,
        'out_relu', (224, 224),
        ['global_average_pooling2d', 'predictions']
    ],
    'Inception-ResNet-V2': [
        tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
        tf.keras.applications.inception_resnet_v2.preprocess_input,
        tf.keras.applications.inception_resnet_v2.decode_predictions,
        'conv_7b_ac', (299, 299),
        ['avg_pool', 'predictions']
    ],
    'Inception-V3': [
        tf.keras.applications.inception_v3.InceptionV3,
        tf.keras.applications.inception_v3.preprocess_input,
        tf.keras.applications.inception_v3.decode_predictions,
        'mixed10', (299, 299),
        ['avg_pool', 'predictions']
    ],
    'Xception': [
        tf.keras.applications.xception.Xception,
        tf.keras.applications.xception.preprocess_input,
        tf.keras.applications.xception.decode_predictions,
        'block14_sepconv2_act', (299, 299),
        ['avg_pool', 'predictions']
    ]
}
