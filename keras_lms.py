import argparse
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback
from tensorflow_large_model_support import LMSKerasCallback

tf.logging.set_verbosity(tf.logging.INFO)

MODELS = {
    "vgg16": tf.keras.applications.VGG16,
    "vgg19": tf.keras.applications.VGG19,
    "inceptionv3": tf.keras.applications.InceptionV3,
    "xception": tf.keras.applications.Xception,
    "resnet50": tf.keras.applications.ResNet50,
    "inceptionresnetv2": tf.keras.applications.InceptionResNetV2,
    "mobilenet": tf.keras.applications.MobileNet,
    "densenet121": tf.keras.applications.DenseNet121,
    "densenet169": tf.keras.applications.DenseNet169,
    "densenet201": tf.keras.applications.DenseNet201,
    "nasnetlarge": tf.keras.applications.NASNetLarge,
    "nasnetmobile": tf.keras.applications.NASNetMobile,
}

DATASETS = {
    "cifar10": (tf.keras.datasets.cifar10, 10),
    "cifar100": (tf.keras.datasets.cifar100, 100),
}

def data_generator(validation_split):
    return tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.0,
	width_shift_range=0.0, height_shift_range=0.0, shear_range=0.0, validation_split=validation_split,
	horizontal_flip=True, fill_mode="nearest")

def get_callbacks(args):
    callbacks = []
    # Enable TFLMS
    if args.lms:
        lms = LMSKerasCallback(n_tensors=args.n_tensors, lb=args.lb,
                               starting_op_names=None)
        callbacks.append(lms)

    return callbacks

def run_model(args):
    # Configure the memory optimizer
    config = tf.ConfigProto()
    config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.SCHEDULING_HEURISTICS
    K.set_session(tf.Session(config=config))

    batch_size = args.batch_size 
    keras_model = MODELS.get(args.model)

    validation_split = args.val_split

    data, num_classes = DATASETS.get(args.dataset)
    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

    input_shape=train_images.shape[1:]
    print(input_shape)

    model = keras_model(weights=None, include_top=True,
                                              input_shape=input_shape,
                                              classes=num_classes)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    ImDataGen = data_generator(validation_split)
    training_data = ImDataGen.flow(train_images, train_labels, batch_size, subset="training")
    validation_data = ImDataGen.flow(train_images, train_labels, batch_size, subset="validation")

    model.fit_generator(training_data, validation_data=validation_data, steps_per_epoch=args.steps,
                           epochs=args.epochs, callbacks=get_callbacks(args))

    model.evaluate(test_images, test_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,
                        default=1,
                        help='Number of epochs to run. (Default 1)')
    parser.add_argument("--steps", type=int,
                        default=10,
                        help='Number of steps per epoch. (Default 10)')
    parser.add_argument("--batch_size", type=int,
                        default=128,
                        help='Number of batch size. (Default 32)')
    parser.add_argument("--val_split", type=float,
                        default=0.2,
                        help='Number of batch size. (Default 0.2)')
    parser.add_argument("--model", type=str,
                        default="resnet50",
                        choices=[i for i in MODELS.keys()],
                        help='model to be benchmarked. (Default: resnet50)')
    parser.add_argument("--dataset", type=str,
                        default="cifar10",
                        choices=[i for i in DATASETS.keys()],
                        help='model to be benchmarked. (Default: cifar10)')

    # LMS parameters
    lms_group = parser.add_mutually_exclusive_group(required=False)
    lms_group.add_argument('--lms', dest='lms', action='store_true',
                           help='Enable TFLMS')
    lms_group.add_argument('--no-lms', dest='lms', action='store_false',
                           help='Disable TFLMS (Default)')
    parser.set_defaults(lms=False)
    parser.add_argument("--n_tensors", type=int,
                        default=-1,
                        help='The number of tensors to swap. Default -1 (all)')
    parser.add_argument("--lb", type=int,
                        default=1,
                        help='Lowerbound value for LMS. A tensor will be '
                             'swapped in during the backward phase at least lb '
                             'nodes before it in the graph. Default 1.')

    args = parser.parse_args()
    run_model(args)
