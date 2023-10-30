# coding=utf-8
import os
import shutil
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import RandomNormal
from layer.layers import Linear

def clean_model(model_dir):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)



def creative_sparse_embedding_dict(sparse_feature_columns, feature_names, embedding_size,
                                   l2_reg_embedding=0.00001, init_std=0.0001, seed=1024):
    sparse_embedding_dict = {feat.name:
                             Embedding(
                                 feat.vocabulary_size,
                                 embedding_size,
                                 embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                 embeddings_regularizer=l2(l2_reg_embedding),
                                 name='emb_' + feat.name
                             )
                        for feat in sparse_feature_columns if feat.name in feature_names}
    return sparse_embedding_dict

def sparse_embedding_lookup(sparse_embedding_dict, sparse_features_input):
    embeddings = []
    for feature_name in sparse_features_input:
        lookup_idx = sparse_features_input[feature_name]
        embeddings.append(sparse_embedding_dict[feature_name](lookup_idx))
    return embeddings


def get_feature_embeddings(sparse_feature_columns, sparse_features_input, embedding_size,
                                   l2_reg_embedding=0.00001, init_std=0.0001, seed=1024):
    sparse_embedding_dict = creative_sparse_embedding_dict(sparse_feature_columns, list(sparse_features_input.keys()), embedding_size,
                                   l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed)
    embeddings = sparse_embedding_lookup(sparse_embedding_dict, sparse_features_input)
    return embeddings


def get_linear_logits(sparse_feature_columns, sparse_features_input, dense_features_input, use_bias=True, l2_reg=0):
    sparse_feature_bias = get_feature_embeddings(sparse_feature_columns, sparse_features_input, embedding_size=1)
    dense_feature_bias = [tf.expand_dims(value, axis=1) for feature, value in dense_features_input.items()]
    if len(sparse_feature_bias)>0 and len(dense_feature_bias)>0:
        sparse_concat = tf.concat(sparse_feature_bias, axis=1)
        dense_concat = tf.concat(dense_feature_bias, axis=1)
        linear_logit = Linear(l2_reg, mode=2, use_bias=use_bias)([sparse_concat, dense_concat])
    elif len(sparse_feature_bias)>0:
        sparse_concat = tf.concat(sparse_feature_bias, axis=1)
        linear_logit = Linear(l2_reg, mode=0, use_bias=use_bias)(sparse_concat)
    elif len(dense_feature_bias)>0:
        dense_concat = tf.concat(dense_feature_bias, axis=1)
        linear_logit = Linear(l2_reg, mode=1, use_bias=use_bias)(dense_concat)
    else:
        raise Exception("No input for linear layer !")
    return linear_logit