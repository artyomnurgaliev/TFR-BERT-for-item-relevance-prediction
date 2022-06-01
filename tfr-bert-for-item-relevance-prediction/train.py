# Copyright 2020 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""An example of using BERT output for finetuning a TF-Ranking model.

Please download `bert_config_file` and `bert_init_ckpt` from tensorflow models
website: https://github.com/tensorflow/models/tree/master/official/nlp/bert.
Note that those checkpoints are TF 2.x compatible, which are different from the
checkpoints downloaded here: https://github.com/google-research/bert. You may
convert a TF 1.x checkpoint to TF 2.x using `tf2_encoder_checkpoint_converter`
under https://github.com/tensorflow/models/tree/master/official/nlp/bert.
The following command downloads an uncased BERT-base model checkpoint for you:

mkdir /tmp/bert && \
wget https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/\
uncased_L-12_H-768_A-12.tar.gz -P /tmp/bert  && \
tar -xvf /tmp/bert/uncased_L-12_H-768_A-12.tar.gz -C /tmp/bert/  && \

Then, use the following command to run training and evaluation locally with cpu
or gpu. For GPU, please add `CUDA_VISIBLE_DEVICES=0` and `--config=cuda`. The
example toy data contains 3 lists in train and test respectively. Due to the
large number of BERT parameters, if running into the `out-of-memory` issue,
plese see: https://github.com/google-research/bert#out-of-memory-issues.

BERT_DIR="/tmp/bert/uncased_L-12_H-768_A-12"  && \
OUTPUT_DIR="/tmp/tfr/model/" && \
DATA_DIR="tensorflow_ranking/extension/testdata" && \
rm -rf "${OUTPUT_DIR}" && \
bazel build -c opt \
tensorflow_ranking/extension/examples:tfrbert_example_py_binary && \
./bazel-bin/tensorflow_ranking/extension/examples/tfrbert_example_py_binary \
   --train_input_pattern=${DATA_DIR}/tfrbert_elwc_train.tfrecord \
   --eval_input_pattern=${DATA_DIR}/tfrbert_elwc_test.tfrecord \
   --bert_config_file=${BERT_DIR}/bert_config.json \
   --bert_init_ckpt=${BERT_DIR}/bert_model.ckpt \
   --bert_max_seq_length=128 \
   --model_dir="${OUTPUT_DIR}" \
   --list_size=3 \
   --loss=softmax_loss \
   --train_batch_size=8 \
   --eval_batch_size=8 \
   --learning_rate=1e-5 \
   --num_train_steps=50 \
   --num_eval_steps=10 \
   --checkpoint_secs=120 \
   --num_checkpoints=20

You can use TensorBoard to display the training results stored in $OUTPUT_DIR.

Notes:
  * Use --alsologtostderr if the output is not printed into screen.
  * The training and evaluation data should be stored in TFRecord format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
from absl import flags
from official.modeling import activations
from official.nlp.bert import configs
from official.nlp.modeling import networks as tfmodel_networks
from tensorflow_ranking.extension.tfrbert import TFRBertUtil
from tensorflow_ranking.python.keras import network as tfrkeras_network

from config import bert_path, max_seq_length

flags.DEFINE_string("config", "cuda", "If 'cuda', running on GPU")

flags.DEFINE_bool("local_training", True, "If true, run training locally.")

flags.DEFINE_list("train_input_pattern", tf.io.gfile.glob('./tfrecords_with_images_128/train???.tfrecords'),
                  "Input file path pattern used for train.")

flags.DEFINE_list("eval_input_pattern", tf.io.gfile.glob('./tfrecords_with_images_128/test???.tfrecords'),
                  "Input file path pattern used for eval.")

flags.DEFINE_string("vocab_input_pattern", "./tfrecords/vocab.txt",
                    "Input file path pattern used for vocab.")

flags.DEFINE_float("learning_rate", 0.0003, "Learning rate for the optimizer.")

batch_size = 1
flags.DEFINE_integer("train_batch_size", batch_size,
                     "Number of input records used per batch for training.")

flags.DEFINE_integer("eval_batch_size", batch_size,
                     "Number of input records used per batch for eval.")

flags.DEFINE_integer("checkpoint_secs", 240,
                     "Saves a model checkpoint every checkpoint_secs seconds.")

flags.DEFINE_integer("num_checkpoints", 100,
                     "Saves at most num_checkpoints checkpoints in workspace.")

flags.DEFINE_integer(
    "num_train_steps", 600000,
    "Number of training iterations. Default means continuous training.")

flags.DEFINE_integer("num_eval_steps", 200, "Number of evaluation iterations.")

flags.DEFINE_string(
    "loss", "approx_ndcg_loss",
    "The RankingLossKey deciding the loss function used in training.")

list_size = 40
flags.DEFINE_integer("list_size", list_size, "List size used for training.")

flags.DEFINE_bool("convert_labels_to_binary", False,
                  "If true, relevance labels are set to either 0 or 1.")

flags.DEFINE_string("model_dir", os.path.normpath("./models_test_3/"), "Output directory for models.")

flags.DEFINE_float("dropout_rate", 0.1, "The dropout rate.")

# The followings are BERT related flags.
flags.DEFINE_string(
    "bert_config_file", bert_path + "bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. This "
    "specifies the model architecture. Please download the model from "
    "the link: https://github.com/google-research/bert")

flags.DEFINE_string(
    "bert_init_ckpt", bert_path + "bert_model.ckpt",
    "Initial checkpoint from a pre-trained BERT model. Please download from "
    "the link: https://github.com/google-research/bert")

flags.DEFINE_integer(
    "bert_max_seq_length", max_seq_length,
    "The maximum input sequence length (#words) after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "bert_num_warmup_steps", 10000,
    "This is used for adjust learning rate. If global_step < num_warmup_steps, "
    "the learning rate will be `global_step/num_warmup_steps * init_lr`. This "
    "is implemented in the bert/optimization.py file.")

FLAGS = flags.FLAGS

_SIZE = "example_list_size"
_NETWORK_NAME = "tfrbert"


def context_feature_columns():
    """Returns context feature names to column definitions."""
    return {}


def example_feature_columns():
    """Returns example feature names to column definitions.

  `input_ids`, `input_mask` and `segment_ids` are derived from query-document
  pair sequence: [CLS] all query tokens [SEP] all document tokens [SEP]. The
  original tokens are mapped to ids (based on BERT vocabulary) in `input_ids`.
  """
    feature_columns = {}
    feature_columns.update({
        "input_ids":
            tf.feature_column.numeric_column(
                "input_ids",
                shape=(FLAGS.bert_max_seq_length,),
                default_value=0,
                dtype=tf.int64),
        "input_mask":
            tf.feature_column.numeric_column(
                "input_mask",
                shape=(FLAGS.bert_max_seq_length,),
                default_value=0,
                dtype=tf.int64),
        "segment_ids":
            tf.feature_column.numeric_column(
                "segment_ids",
                shape=(FLAGS.bert_max_seq_length,),
                default_value=0,
                dtype=tf.int64),
        "item_price":
            tf.feature_column.numeric_column(
                "item_price",
                shape=(1,),
                default_value=0,
                dtype=tf.float32),
        'image_shape':
            tf.feature_column.numeric_column(
                'image_shape',
                shape=(1,),
                default_value=0,
                dtype=tf.int64),
        'image_count':
            tf.feature_column.numeric_column(
                'image_count',
                shape=(1,),
                default_value=0,
                dtype=tf.int64),
        'image':
            tf.feature_column.numeric_column(
                "image",
                shape=(40 * 40 * 3,),
                default_value=0,
                dtype=tf.float32),
    })
    return feature_columns


class TFRBertRankingNetwork(tfrkeras_network.UnivariateRankingNetwork):
    """A TFRBertRankingNetwork scoring based univariate ranking network."""

    def __init__(self,
                 context_feature_columns,
                 example_feature_columns,
                 bert_config_file,
                 bert_max_seq_length,
                 bert_output_dropout,
                 name="tfrbert",
                 **kwargs):
        """Initializes an instance of TFRBertRankingNetwork.

    Args:
      context_feature_columns: A dict containing all the context feature columns
        used by the network. Keys are feature names, and values are instances of
        classes derived from `_FeatureColumn`.
      example_feature_columns: A dict containing all the example feature columns
        used by the network. Keys are feature names, and values are instances of
        classes derived from `_FeatureColumn`.
      bert_config_file: (string) path to Bert configuration file.
      bert_max_seq_length: (int) maximum input sequence length (#words) after
        WordPiece tokenization. Sequences longer than this will be truncated,
        and shorter than this will be padded.
      bert_output_dropout: When not `None`, the probability will be used as the
        dropout probability for BERT output.
      name: name of Keras network.
      **kwargs: keyword arguments.
    """
        super(TFRBertRankingNetwork, self).__init__(
            context_feature_columns=context_feature_columns,
            example_feature_columns=example_feature_columns,
            name=name,
            **kwargs)

        self._bert_config_file = bert_config_file
        self._bert_max_seq_length = bert_max_seq_length
        self._bert_output_dropout = bert_output_dropout

        bert_config = configs.BertConfig.from_json_file(self._bert_config_file)
        self._bert_encoder = tfmodel_networks.TransformerEncoder(
            vocab_size=bert_config.vocab_size,
            hidden_size=bert_config.hidden_size,
            num_layers=bert_config.num_hidden_layers,
            num_attention_heads=bert_config.num_attention_heads,
            intermediate_size=bert_config.intermediate_size,
            activation=activations.gelu,
            dropout_rate=bert_config.hidden_dropout_prob,
            attention_dropout_rate=bert_config.attention_probs_dropout_prob,
            sequence_length=self._bert_max_seq_length,
            max_sequence_length=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range))

        self._dropout_layer = tf.keras.layers.Dropout(
            rate=self._bert_output_dropout)

        self._conv1 = tf.keras.layers.Conv2D(10, 3, activation='relu')
        self._conv2 = tf.keras.layers.Conv2D(10, 3, activation='relu')
        self._maxpool = tf.keras.layers.MaxPooling2D()
        self._flatten = tf.keras.layers.Flatten()
        self._dense1 = tf.keras.layers.Dense(1, activation='relu')
        self._dense2 = tf.keras.layers.Dense(4, activation='relu')
        self._dense3 = tf.keras.layers.Dense(32, activation='relu')
        self._dense4 = tf.keras.layers.Dense(16, activation='relu')
        self._score_layer = tf.keras.layers.Dense(units=1, name="score")

    def score(self, context_features=None, example_features=None, training=True):
        """Univariate scoring of context and one example to generate a score.

    Args:
      context_features: (dict) Context feature names to 2D tensors of shape
        [batch_size, ...].
      example_features: (dict) Example feature names to 2D tensors of shape
        [batch_size, ...].
      training: (bool) Whether in training or inference mode.

    Returns:
      (tf.Tensor) A score tensor of shape [batch_size, 1].
    """

        # @tf.function
        def get_inputs():
            inputs = {
                "input_word_ids": tf.cast(example_features["input_ids"], tf.int32),
                "input_mask": tf.cast(example_features["input_mask"], tf.int32),
                "input_type_ids": tf.cast(example_features["segment_ids"], tf.int32)
            }
            # print(f"input ids shape: {example_features['input_ids'].shape}")

            item_price = example_features["item_price"]
            image_count = example_features["image_count"]
            image_shape = example_features["image_shape"]
            image = example_features["image"]

            img = tf.reshape(image, [tf.shape(image)[0], 40, 40, 3])
            img = self._conv1(img)
            img = self._maxpool(img)
            img = self._conv2(img)
            img = self._maxpool(img)
            img = self._flatten(img)
            img = self._dense1(img) # (40, 1)

            output = tf.concat([item_price] + [image_count] + [image_shape] + [img], axis=1)
            output = self._dense2(output, training=training)

            # The `bert_encoder` returns a tuple of (sequence_output, cls_output).
            _, cls_output = self._bert_encoder(inputs, training=training)

            result = tf.concat([output] + [cls_output], axis=1)
            return result

        result = get_inputs()

        output = self._dropout_layer(result, training=training)
        output = self._dense3(output)
        output = self._dense4(output)
        return self._score_layer(output)

    def get_config(self):
        config = super(TFRBertRankingNetwork, self).get_config()
        config.update({
            "bert_config_file": self._bert_config_file,
            "bert_max_seq_length": self._bert_max_seq_length,
            "bert_output_dropout": self._bert_output_dropout,
        })

        return config


def get_estimator(hparams):
    """Create Keras ranking estimator."""

    util = TFRBertUtil(
        bert_config_file=hparams.get("bert_config_file"),
        bert_init_ckpt=hparams.get("bert_init_ckpt"),
        bert_max_seq_length=hparams.get("bert_max_seq_length"))

    network = TFRBertRankingNetwork(
        context_feature_columns=context_feature_columns(),
        example_feature_columns=example_feature_columns(),
        bert_config_file=hparams.get("bert_config_file"),
        bert_max_seq_length=hparams.get("bert_max_seq_length"),
        bert_output_dropout=hparams.get("dropout_rate"),
        name=_NETWORK_NAME)

    loss = tfr.keras.losses.get(
        hparams.get("loss"),
        reduction=tf.compat.v2.losses.Reduction.SUM_OVER_BATCH_SIZE)

    metrics = tfr.keras.metrics.default_keras_metrics()

    config = tf.estimator.RunConfig(
        model_dir=hparams.get("model_dir"),
        keep_checkpoint_max=hparams.get("num_checkpoints"),
        save_checkpoints_secs=hparams.get("checkpoint_secs"))

    optimizer = util.create_optimizer(
        init_lr=hparams.get("learning_rate"),
        train_steps=hparams.get("num_train_steps"),
        warmup_steps=hparams.get("bert_num_warmup_steps"))

    ranker = tfr.keras.model.create_keras_model(
        network=network,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        size_feature_name=_SIZE)

    return tfr.keras.estimator.model_to_estimator(
        model=ranker,
        model_dir=hparams.get("model_dir"),
        config=config,
        warm_start_from=util.get_warm_start_settings(exclude=_NETWORK_NAME))


def train_and_eval():
    """Runs the training and evaluation jobs for a BERT ranking model."""
    # The below contains a set of common flags for a TF-Ranking model. You need to
    # include all of them for adopting the `RankingPipeline`.
    hparams = dict(
        train_input_pattern=FLAGS.train_input_pattern,
        eval_input_pattern=FLAGS.eval_input_pattern,
        learning_rate=FLAGS.learning_rate,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        checkpoint_secs=FLAGS.checkpoint_secs,
        num_checkpoints=FLAGS.num_checkpoints,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        loss=FLAGS.loss,
        dropout_rate=FLAGS.dropout_rate,
        list_size=FLAGS.list_size,
        listwise_inference=True,  # Only supports `True` in keras Ranking Network.
        convert_labels_to_binary=FLAGS.convert_labels_to_binary,
        model_dir=FLAGS.model_dir,
        bert_config_file=FLAGS.bert_config_file,
        bert_init_ckpt=FLAGS.bert_init_ckpt,
        bert_max_seq_length=FLAGS.bert_max_seq_length,
        bert_num_warmup_steps=FLAGS.bert_num_warmup_steps,
        config=FLAGS.config,
    )

    bert_ranking_pipeline = tfr.ext.pipeline.RankingPipeline(
        context_feature_columns=context_feature_columns(),
        example_feature_columns=example_feature_columns(),
        hparams=hparams,
        estimator=get_estimator(hparams),
        label_feature_name="rank",
        label_feature_type=tf.float32,
        size_feature_name=_SIZE)

    bert_ranking_pipeline.train_and_eval(local_training=FLAGS.local_training)


def main(_):
    tf.debugging.set_log_device_placement(True)

    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)

    train_and_eval()


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(tf.config.list_physical_devices('GPU'))
    print(tf.test.is_gpu_available())
    print(tf.test.is_built_with_cuda())
    tf.compat.v1.app.run()
