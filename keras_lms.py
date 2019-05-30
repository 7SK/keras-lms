import argparse
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback

from official.resnet import imagenet_main
from official.utils.misc import distribution_utils
from tensorflow_large_model_support import LMSKerasCallback
import os

tf.logging.set_verbosity(tf.logging.INFO)

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

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

def get_callbacks(args):
    callbacks = []
    # Enable TFLMS
    if args.lms:
        lms = LMSKerasCallback(n_tensors=args.n_tensors, lb=args.lb,
                               starting_op_names=None)
        callbacks.append(lms)
    return callbacks

def data_pipeline(data_dir, batch_size, is_training, dtype, num_epochs):
    if is_training:
        filenames = [os.path.join(data_dir, 'train-%05d-of-01024' % i)
                    for i in range(1024)]
        dataset = tf.data.Dataset.from_tensor_slices(filenames).shuffle(buffer_size=1024)
    else:
        filenames = [os.path.join(data_dir, 'validation-%05d-of-00128' % i)
                    for i in range(128)]
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
#    if input_context:
#        tf.compat.v1.logging.info(
#            'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
#                input_context.input_pipeline_id, input_context.num_input_pipelines))
#        dataset = dataset.shard(input_context.num_input_pipelines,
#                            input_context.input_pipeline_id)

    dataset = dataset.interleave(
              tf.data.TFRecordDataset,
              cycle_length=10,
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(
              lambda value: parse_tfrecords(value, is_training, dtype),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def parse_tfrecords(raw_record, is_training, dtype):
    image, label = imagenet_main.parse_record(raw_record, is_training, dtype)
    label = tf.cast(tf.cast(tf.reshape(label, shape=[1]), dtype=tf.int32) - 1,
            dtype=tf.float32)
    return image,label

def run_model(args):
    # Configure the memory optimizer
    config = tf.ConfigProto()
    config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.SCHEDULING_HEURISTICS
    K.set_session(tf.Session(config=config))

    data_dir = args.data_dir 
    val_freq = args.val_freq
    train_epochs = args.num_epoch
    val_epochs = 1
    keras_model = MODELS.get(args.model)
    train_steps = NUM_IMAGES['train'] // args.batch_size
    val_steps = NUM_IMAGES['validation'] // args.batch_size

#    strategy = distribution_utils.get_distribution_strategy(
#    distribution_strategy=args.distribution_strategy,
#    num_gpus=args.num_gpus,
#    num_workers=distribution_utils.configure_cluster(),
#    all_reduce_alg=args.all_reduce_alg,
#    num_packs=args.num_packs)

#    strategy_scope = distribution_utils.get_strategy_scope(strategy)

    train_data = data_pipeline(data_dir, args.batch_size, True, float, train_epochs)
    val_data = data_pipeline(data_dir, args.batch_size, False, float, val_epochs)

    model = keras_model(weights=None, include_top=True,
                        input_shape=[224,224,3], classes=1000)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    model.fit(train_data,
              epochs=train_epochs,
              steps_per_epoch=train_steps,
              callbacks=get_callbacks(args),
              validation_steps=val_steps,
              validation_data=val_data,
              validation_freq=val_freq,
              verbose=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int,
                        default=1,
                        help='Number of epochs to run. (Default 1)')
    parser.add_argument("--batch_size", type=int,
                        default=64,
                        help='Number of batch size. (Default 64)')
    parser.add_argument("--num_gpu", type=int,
                        default=1,
                        help='Number of gpu. (Default 1)')
    parser.add_argument("--val_freq", type=int,
                        default=16,
                        help='epochs between evals. (Default 16)')
    parser.add_argument("--model", type=str,
                        default="resnet50",
                        choices=[i for i in MODELS.keys()],
                        help='model to be benchmarked. (Default: resnet50)')
    parser.add_argument("--data_dir", type=str,
                        default=None,
                        help='data dir path. (Default: None)')

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
