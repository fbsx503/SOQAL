# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf


class BERT_model:
    def __init__(self, vocab, output):
        self.vocab_file = vocab
        self.output_dir = output
        self.init_checkpoint = output
        tf.logging.set_verbosity(0)
        self.bert_config = modeling.BertConfig(vocab_size=64000, hidden_size=512, num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
        tf.gfile.MakeDirs(self.output_dir)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=do_lower_case)
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        self.run_config = tf.contrib.tpu.RunConfig(
            cluster=None,
            master=master,
            model_dir=self.output_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=iterations_per_loop,
                num_shards=num_tpu_cores,
                per_host_input_for_training=is_per_host))
        num_train_steps = None
        num_warmup_steps = None
        self.model_fn = model_fn_builder(
            bert_config=self.bert_config,
            init_checkpoint=self.init_checkpoint,
            learning_rate=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=use_tpu,
            use_one_hot_embeddings=use_tpu)
        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=use_tpu,
            model_fn=self.model_fn,
            config=self.run_config,
            train_batch_size=train_batch_size,
            predict_batch_size=predict_batch_size)

    def input_to_squadformat(self, P, Q):
        id_i = 0
        qas = []
        id = str(id_i)
        ques = Q
        ans = ""
        answer_start = 0
        answer = {
            'text': ans,
            'answer_start': answer_start
        }
        question = {
            'question': ques,
            'id': id,
            'answers': [answer]
        }
        qas.append(question)
        paragraph = {
            'context': P,
            'qas': qas
        }
        article = {
            'title': Q,
            'paragraphs': [paragraph]
        }
        return [article]

    def predict_batch(self, input_data):
        eval_examples = read_squad_examples_input(input_data)
        # eval_examples = read_squad_examples(
        #  input_file=predict_file, is_training=False)

        eval_writer = FeatureWriter(
            filename=os.path.join(self.output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", predict_batch_size)

        all_results = []

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=False)

        all_results = []
        for result in self.estimator.predict(
                predict_input_fn, yield_single_examples=True):
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

        nbest, pred = write_predictions(eval_examples, eval_features, all_results,
                                        n_best_size, max_answer_length,
                                        do_lower_case, None,
                                        None, None)
        return nbest

    def predict_example(self, P, Q):
        file = self.input_to_squadformat(P, Q)
        eval_examples = read_squad_examples_input(file)
        eval_writer = FeatureWriter(
            filename=os.path.join(self.output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions on single example *****")

        all_results = []

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=False)

        all_results = []
        for result in self.estimator.predict(
                predict_input_fn, yield_single_examples=True):
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

        nbest, pred = write_predictions(eval_examples, eval_features, all_results,
                                        n_best_size, max_answer_length,
                                        do_lower_case, None,
                                        None, None)
        return pred['0']
