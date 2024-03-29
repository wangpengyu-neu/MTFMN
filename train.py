# coding=utf-8

import tensorflow as tf
from util import input_util, model_util
from model.mmoe import model_fn
from util.input_util import input_fn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("model_dir", 'save_model/MMOE', 'model_dir')
tf.app.flags.DEFINE_integer("train", 1, 'train mode')
tf.app.flags.DEFINE_integer("eval", 1, 'eval mode')
tf.app.flags.DEFINE_integer("epoch", 1, 'epoch')
tf.app.flags.DEFINE_integer("batch_size", 256, 'batch_size')


def main(_):


    path = "data/cenc.data.gz"

    sparse_features, dense_features, sparse_feature_columns, dense_feature_columns, label = input_util.load_cenc_data(path, sep=',')

    train_size = int(len(label) * 0.85)
    train_sparse_features = sparse_features[:train_size]
    train_dense_features = dense_features[:train_size]
    train_label = label[:train_size]
    eval_sparse_features = sparse_features[train_size:]
    eval_dense_features = dense_features[train_size:]
    eval_label = label[train_size:]

    model_params = {
        "sparse_feature_columns": sparse_feature_columns,
        "dense_feature_columns": dense_feature_columns,
        "batch_size": FLAGS.batch_size,
        "embedding_size": 8,
        "task_nums": 2,
        "lr": 0.01
    }

    model_util.clean_model(FLAGS.model_dir)

    config = tf.estimator.RunConfig()
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.train == 1:
        estimator.train(
            input_fn=lambda: input_fn(train_sparse_features, train_dense_features, train_label, epoch=FLAGS.epoch,
                                      batch_size=FLAGS.batch_size, if_multi_label=True)
        )

    if FLAGS.eval == 1:
        estimator.evaluate(
            input_fn=lambda: input_fn(eval_sparse_features, eval_dense_features, eval_label, epoch=FLAGS.epoch,
                                      batch_size=FLAGS.batch_size, if_multi_label=True)
        )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)