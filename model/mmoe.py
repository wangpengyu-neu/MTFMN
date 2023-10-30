# coding=utf-8

import tensorflow as tf
from layer import layers
from metrics import metrics
from util import model_util

def model_fn(features, labels, mode, params):
    sparse_features_input = features['sparse_features']
    batch_size = params['batch_size']
    lr = params['lr']
    embedding_size = params['embedding_size']
    task_nums = params['task_nums']
    sparse_feature_columns = params['sparse_feature_columns']
    sparse_embeddings = model_util.get_feature_embeddings(
        sparse_feature_columns,
        dict([(feature_name, input) for feature_name, input in sparse_features_input.items()]),
        embedding_size=embedding_size
    )
    concat_embeddings = tf.concat(sparse_embeddings, axis=1)
    mmoe_layers = layers.MMoE(units=4, num_experts=8, num_tasks=task_nums)(concat_embeddings)
    logits = []
    for index, task_layer in enumerate(mmoe_layers):

        tower_layer = tf.keras.layers.Dense(units=8, activation='relu',
                                   kernel_initializer=tf.keras.initializers.VarianceScaling())(task_layer)

        # batch_size * 1
        output_layer = tf.keras.layers.Dense(units=1, activation=None,
                                    kernel_initializer=tf.keras.initializers.VarianceScaling())(tower_layer)

        logits.append(tf.squeeze(output_layer, axis=1))

    labels = [labels['income_50k'], labels['marital_stat']]

    preds = [tf.sigmoid(logit) for logit in logits]

    predictions = {
        "prob_income_50k": preds[0],
        "prob_marital_stat": preds[1]
    }

    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    else:
        losses = 0
        for label, logit in zip(labels, logits):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
            loss = tf.reduce_sum(loss)
            losses += loss

        if mode == tf.estimator.ModeKeys.EVAL:

            eval_hooks = [
                metrics.BinaryMetricHook(batch_size=batch_size, pred_tensor=pred, label_tensor=label, prefix='Eval_{}'.format(index))
                for index, (label, pred) in enumerate(zip(labels, preds))
            ]

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=losses,
                evaluation_hooks=eval_hooks
            )

        if mode == tf.estimator.ModeKeys.TRAIN:

            training_hooks = [
                metrics.BinaryMetricHook(batch_size=batch_size, pred_tensor=pred, label_tensor=label,
                                         prefix='Train_{}'.format(index))
                for index, (label, pred) in enumerate(zip(labels, preds))
            ]

            optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

            train_op = optimizer.minimize(losses, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=losses,
                train_op=train_op,
                training_hooks=training_hooks
            )