from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import shutil

import tensorflow.compat.v1 as tf

from .electra import configure_finetuning
from .electra.finetune import preprocessing
from .electra.finetune import task_builder
from .electra.model import modeling
from .electra.model import optimization
from .electra.model import tokenization
from .electra.util import training_utils
from .electra.util import utils
from .electra.finetune.qa.qa_tasks import QAExample

tf.logging.set_verbosity(tf.logging.ERROR)

DATA_MODEL_DIR = '/mnt/427AB1F27AB1E339/CurrentSemester/NeuralArabicQuestionAnswering/DownloadedForGP/tfmodel/model/'
INIT_CHECKPOINT = DATA_MODEL_DIR + 'model/model.ckpt-957'

class FinetuningModel(object):
    """Finetuning model with support for multi-task training."""

    def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
                 is_training, features, num_train_steps):
        # Create a shared transformer encoder
        bert_config = training_utils.get_bert_config(config)
        self.bert_config = bert_config
        if config.debug:
            bert_config.num_hidden_layers = 3
            bert_config.hidden_size = 144
            bert_config.intermediate_size = 144 * 4
            bert_config.num_attention_heads = 4
        assert config.max_seq_length <= bert_config.max_position_embeddings
        bert_model = modeling.BertModel(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=features["input_ids"],
            input_mask=features["input_mask"],
            token_type_ids=features["segment_ids"],
            use_one_hot_embeddings=config.use_tpu,
            embedding_size=config.embedding_size)
        percent_done = (tf.cast(tf.train.get_or_create_global_step(), tf.float32) /
                        tf.cast(num_train_steps, tf.float32))

        # Add specific tasks
        self.outputs = {"task_id": features["task_id"]}
        losses = []
        for task in tasks:
            with tf.variable_scope("task_specific/" + task.name):
                task_losses, task_outputs = task.get_prediction_module(
                    bert_model, features, is_training, percent_done)
                losses.append(task_losses)
                self.outputs[task.name] = task_outputs
        self.loss = tf.reduce_sum(
            tf.stack(losses, -1) *
            tf.one_hot(features["task_id"], len(config.task_names)))


def model_fn_builder(config: configure_finetuning.FinetuningConfig, tasks,
                     num_train_steps, pretraining_config=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    utils.log("Building model...")
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = FinetuningModel(
        config, tasks, is_training, features, num_train_steps)

    # Load pre-trained weights from checkpoint
    init_checkpoint = config.init_checkpoint
    if pretraining_config is not None:
      init_checkpoint = pretraining_config['checkpoint']
      utils.log("Using checkpoint", init_checkpoint)
    tvars = tf.trainable_variables()
    print("\n\n\n\n")
    print(tvars)
    print("\n\n\n\n")
    scaffold_fn = None
    if init_checkpoint:
      assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
          tvars, init_checkpoint)
      if config.use_tpu:
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()
        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # Build model for training or prediction
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          model.loss, config.learning_rate, num_train_steps,
          weight_decay_rate=config.weight_decay_rate,
          use_tpu=config.use_tpu,
          warmup_proportion=config.warmup_proportion,
          layerwise_lr_decay_power=config.layerwise_lr_decay,
          n_transformer_layers=model.bert_config.num_hidden_layers
      )
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=model.loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          training_hooks=[training_utils.ETAHook(
              {} if config.use_tpu else dict(loss=model.loss),
              num_train_steps, config.iterations_per_loop, config.use_tpu, 10)])
    else:
      assert mode == tf.estimator.ModeKeys.PREDICT
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=utils.flatten_dict(model.outputs),
          scaffold_fn=scaffold_fn)

    utils.log("Building complete")
    return output_spec

  return model_fn

class ModelRunner(object):
    """Fine-tunes a model on a supervised task."""

    def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
                 pretraining_config=None):
        self._config = config
        self._tasks = tasks
        self._preprocessor = preprocessing.Preprocessor(config, self._tasks)

        is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
        tpu_cluster_resolver = None

        if config.use_tpu and config.tpu_name:
            tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                config.tpu_name, zone=config.tpu_zone, project=config.gcp_project)
        tpu_config = tf.estimator.tpu.TPUConfig(
            iterations_per_loop=config.iterations_per_loop,
            num_shards=config.num_tpu_cores,
            per_host_input_for_training=is_per_host,
            tpu_job_name=config.tpu_job_name)

        run_config = tf.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=config.model_dir,
            save_checkpoints_steps=config.save_checkpoints_steps,
            save_checkpoints_secs=None,
            tpu_config=tpu_config)

        self._train_input_fn, self.train_steps = None, 0

        model_fn = model_fn_builder(
            config,
            self._tasks,
            self.train_steps,
            pretraining_config)

        self._estimator = tf.estimator.tpu.TPUEstimator(
            use_tpu=config.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            predict_batch_size=config.predict_batch_size)


    def predict(self, split = "dev"):
        task = self._tasks[0]
        eval_input_fn, _ = self._preprocessor.prepare_predict([task], split)
        results = self._estimator.predict(input_fn=eval_input_fn, yield_single_examples=True)
        print("results are ")
        print(results)
        print("\n\n\n\n\n\n")
        var_name_list = [v.name for v in tf.trainable_variables()]
        print(var_name_list)
        scorer = task.get_scorer()
        for r in results:
            if r["task_id"] != len(self._tasks):  # ignore padding examples
                r = utils.nest_dict(r, self._config.task_names)
                scorer.update(r[task.name])
        return scorer.get_results()


def init_model():
    data_dir = DATA_MODEL_DIR + 'data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    hparams = dict()
    config = configure_finetuning.FinetuningConfig('tf-araelectra-base', DATA_MODEL_DIR, **hparams)
    tasks = task_builder.get_tasks(config)

    pretraining_config = {'checkpoint': INIT_CHECKPOINT}
    model_runner = ModelRunner(config, tasks, pretraining_config=pretraining_config)
    return model_runner


def predict(dataset, model):
    model._tasks[0]._examples = {}
    try:
        os.remove(DATA_MODEL_DIR + 'data/dev.json')
        shutil.rmtree(DATA_MODEL_DIR + 'tfrecords')
    except Exception as err:
        print(err)

    with open(DATA_MODEL_DIR + 'data/dev.json', 'w') as f:
        f.writelines(json.dumps(dataset))
        f.close()

    from tensorflow.python import pywrap_tensorflow
    print("CHECKPOINT Baby")
    ckpt_path = "/mnt/427AB1F27AB1E339/CurrentSemester/NeuralArabicQuestionAnswering/DownloadedForGP/tfmodel/model/model/finetuning_models/squad_model/model.ckpt-957"
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print(var_to_shape_map)

    return model.predict()



