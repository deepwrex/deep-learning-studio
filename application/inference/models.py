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
    ],
    'DenseNet121': [
        tf.keras.applications.densenet.DenseNet121,
        tf.keras.applications.densenet.preprocess_input,
        tf.keras.applications.densenet.decode_predictions,
        'relu', (299, 299), ['avg_pool', 'predictions']
    ],
    'DenseNet169': [
        tf.keras.applications.densenet.DenseNet169,
        tf.keras.applications.densenet.preprocess_input,
        tf.keras.applications.densenet.decode_predictions,
        'relu', (299, 299), ['avg_pool', 'predictions']
    ],
    'DenseNet201': [
        tf.keras.applications.densenet.DenseNet201,
        tf.keras.applications.densenet.preprocess_input,
        tf.keras.applications.densenet.decode_predictions,
        'relu', (299, 299), ['avg_pool', 'predictions']
    ],
    'ResNet50-V2': [
        tf.keras.applications.resnet_v2.ResNet50V2,
        tf.keras.applications.resnet_v2.preprocess_input,
        tf.keras.applications.resnet_v2.decode_predictions,
        'post_relu', (299, 299), ['avg_pool', 'predictions']
    ],
    'ResNet101-V2': [
        tf.keras.applications.resnet_v2.ResNet101V2,
        tf.keras.applications.resnet_v2.preprocess_input,
        tf.keras.applications.resnet_v2.decode_predictions,
        'post_relu', (299, 299), ['avg_pool', 'predictions']
    ],
    'ResNet152-V2': [
        tf.keras.applications.resnet_v2.ResNet152V2,
        tf.keras.applications.resnet_v2.preprocess_input,
        tf.keras.applications.resnet_v2.decode_predictions,
        'post_relu', (299, 299), ['avg_pool', 'predictions']
    ],
    'EfficientNetB0': [
        tf.keras.applications.efficientnet.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
        tf.keras.applications.efficientnet.decode_predictions,
        'top_activation', (299, 299),
        ['avg_pool', 'top_dropout', 'predictions']
    ],
    'EfficientNetB1': [
        tf.keras.applications.efficientnet.EfficientNetB1,
        tf.keras.applications.efficientnet.preprocess_input,
        tf.keras.applications.efficientnet.decode_predictions,
        'top_activation', (299, 299),
        ['avg_pool', 'top_dropout', 'predictions']
    ],
    'EfficientNetB2': [
        tf.keras.applications.efficientnet.EfficientNetB2,
        tf.keras.applications.efficientnet.preprocess_input,
        tf.keras.applications.efficientnet.decode_predictions,
        'top_activation', (299, 299),
        ['avg_pool', 'top_dropout', 'predictions']
    ],
    'EfficientNetB3': [
        tf.keras.applications.efficientnet.EfficientNetB3,
        tf.keras.applications.efficientnet.preprocess_input,
        tf.keras.applications.efficientnet.decode_predictions,
        'top_activation', (299, 299),
        ['avg_pool', 'top_dropout', 'predictions']
    ],
    'EfficientNetB4': [
        tf.keras.applications.efficientnet.EfficientNetB4,
        tf.keras.applications.efficientnet.preprocess_input,
        tf.keras.applications.efficientnet.decode_predictions,
        'top_activation', (299, 299),
        ['avg_pool', 'top_dropout', 'predictions']
    ],
    'EfficientNetB5': [
        tf.keras.applications.efficientnet.EfficientNetB5,
        tf.keras.applications.efficientnet.preprocess_input,
        tf.keras.applications.efficientnet.decode_predictions,
        'top_activation', (299, 299),
        ['avg_pool', 'top_dropout', 'predictions']
    ],
    'EfficientNetB6': [
        tf.keras.applications.efficientnet.EfficientNetB6,
        tf.keras.applications.efficientnet.preprocess_input,
        tf.keras.applications.efficientnet.decode_predictions,
        'top_activation', (299, 299),
        ['avg_pool', 'top_dropout', 'predictions']
    ],
    'EfficientNetB7': [
        tf.keras.applications.efficientnet.EfficientNetB7,
        tf.keras.applications.efficientnet.preprocess_input,
        tf.keras.applications.efficientnet.decode_predictions,
        'top_activation', (299, 299),
        ['avg_pool', 'top_dropout', 'predictions']
    ],
}
