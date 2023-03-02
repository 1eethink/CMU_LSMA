#!/bin/python

from http.client import UnimplementedFileMode
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import sys
import pdb


# Train SVM

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':
  args = parser.parse_args()

  fread = open(args.list_videos, "r")
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category

  feat_list = []
  label_list = []

  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignore
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))
  #1. Train a SVM classifier using feat_list and label_list
  # below are the initial settings you could use
  # cache_size=2000, decision_function_shape='ovr', kernel="rbf"
  # your model should be named as "clf" to match the variable in pickle.dump()

  X_train, X_test, y_train, y_test = train_test_split(feat_list, label_list, test_size=0.15, random_state=42)

  clf = make_pipeline(StandardScaler(), SVC(cache_size=3000, decision_function_shape='ovr',kernel='rbf', probability=True)).fit(X_train, y_train)
  # clf = SVC(cache_size=1000, decision_function_shape='ovr',kernel='poly', degree=3).fit(X_train, y_train)

  print("train accuracy: ", clf.score(X_train, y_train))
  print("test accuracy: ", clf.score(X_test, y_test))


  # raise NotImplemented("Please fill the blank")

  # save trained SVM in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('One-versus-rest multi-class SVM trained successfully')
