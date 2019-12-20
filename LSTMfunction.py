# -*- coding: utf-8 -*-

import os
import re
import string
import requests
import numpy as np
import collections
import random
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()

# セッション
sess = tf.Session()

# RNNの設定
num_layers = 3              # RNN層の数
min_word_freq = 5           # 出現頻度がこの値以下の単語の除外
rnn_size = 128              # RNNモデルのサイズ
epochs = 10                 # データを処理する回数
batch_size = 100            # 一度にトレーニングするサンプル数
learning_rate = 0.0005      # 学習率
training_seq_len = 50       # 前後（左右）の単語の数（左右に２５単語ずつ）
save_every = 500            # モデルを保存する頻度
eval_every = 50             # テスト文を評価する頻度

