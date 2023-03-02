#!/bin/python

import argparse
import os
import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import sys

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category

  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))


  #1. Train a MLP classifier using feat_list and label_list
  # below are the initial settings you could use
  # hidden_layer_sizes=(512),activation="relu",solver="adam",alpha=1e-3
  # your model should be named as "clf" to match the variable in pickle.dump()
  print('MLP classifier training start!')
  X_train, X_test, y_train, y_test = train_test_split(feat_list, label_list, test_size=0.1, random_state=42)
  # clf = make_pipeline(StandardScaler(), SVC(cache_size=1000, decision_function_shape='ovr',kernel='rbf', probability=True)).fit(X_train, y_train)

  clf = MLPClassifier(hidden_layer_sizes=(700, 256), batch_size = 256, activation='relu',solver='adam', learning_rate_init=0.001,learning_rate='adaptive', alpha=1e-4, early_stopping=True, validation_fraction=0.2, verbose=True).fit(X_train,y_train)
  print("train accuracy: ", clf.score(X_train, y_train))
  print("test accuracy: ", clf.score(X_test, y_test))

  # clf = make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(512, 512), batch_size = 256, activation='relu',solver='adam', learning_rate='adaptive',alpha=1e-4, early_stopping=True)).fit(X_train, y_train)
  # print("train accuracy: ", clf.score(X_train, y_train))
  # print("test accuracy: ", clf.score(X_test, y_test))

  # clf = make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(512, 512), batch_size = 256, activation='relu',solver='adam', learning_rate='adaptive',alpha=1e-2, early_stopping=True)).fit(X_train, y_train)
  # print("train accuracy: ", clf.score(X_train, y_train))
  # print("test accuracy: ", clf.score(X_test, y_test))


  # raise NotImplementedError("Fill in the blank first")
  # save trained MLP in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')
