# CMU 11-775 Fall 2022 Homework 2

This Homework is done by Hojeong Lee.

## Install Dependencies & Dataset

From installing dependencies and dataset, I just followed the descriptions.

Eventually, the directory structure should look like this:

* this repo
  * code
  * data
    * videos (unzipped from 11775_s22_data.zip)
    * labels (unzipped from 11775_s22_data.zip)
  * env
  * ...


## SIFT Features

To extract SIFT features, I use the command:

```bash
python code/run_sift.py data/labels/train_val.csv
```

and for the test set:

```bash
python code/run_sift.py data/labels/test_for_students.csv
```



To train K-Means with SIFT feature for 128 clusters, I use:

```bash
python code/train_kmeans.py data/labels/train_val.csv data/sift 128 sift_128
```



To extract Bag-of-Words representation with the trained model, I use:

```bash
python code/run_bow.py data/labels/train_val.csv sift_128 data/sift
```

and for the test set:

```bash
python code/run_bow.py data/labels/test_for_students.csv sift_128 data/sift
```


## CNN Features

To extract CNN features, I use:

```bash
python code/run_cnn.py data/labels/train_val.csv
```

and for test set,

```bash
python code/run_cnn.py data/labels/test_for_students.csv
```

## 3D CNN Features

To extract 3D CNN features, I use:

```bash
python code/run_cnn3d.py data/labels/train_val.csv
```

and for test set,

```bash
python code/run_cnn3d.py data/labels/test_for_students.csv
```


## MLP Classifier


To train MLP with SIFT Bag-of-Words, I use:

```bash
python code/run_mlp.py sift --feature_dir data/bow_sift_128 --num_features 128
```

To train MLP with CNN features, I use:

```bash
python code/run_mlp.py cnn --feature_dir data/cnn --num_features 512
```

To train MLP with 3d CNN features, I use:

```bash
python code/run_mlp.py cnn3d --feature_dir data/cnn3d --num_features 512
```