
from bert import modeling
import tensorflow as tf
import numpy as np


BERT_CONFIG = 'uncased_L-12_H-768_A-12/bert_config.json'

bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)

class Model:
    def __init__(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        model=modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=self.X,
            use_one_hot_embeddings=False
        )
        output_layer=model.get_sequence_output()
        embedding=model.get_embedding_table()
        with tf.variable_scope('cls/predictions'):
            with tf.variable_scope('transform'):
                input_tensor = tf.layers.dense(
                    output_layer,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range
                    ),
                )
                input_tensor = modeling.layer_norm(input_tensor)

            output_bias = tf.get_variable(
                'output_bias',
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer(),
            )
            logits = tf.matmul(input_tensor, embedding, transpose_b=True)
            self.logits = tf.nn.bias_add(logits, output_bias)
