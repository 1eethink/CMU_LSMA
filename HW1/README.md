## Environment Settings

I followed the given commands on github and handout.

## Task 1: MFCC-Bag-Of-Features

I just followed the provided guidelines.

## Task 2: SVM classifier & MLP classifier

### SVM classifier

I can train the model through:
```
$ python train_svm_multiclass.py bof/ 50 labels/train_val.csv weights/mfcc-50.svm.model
```
Then, I can get the prediction through:
```
$ python test_svm_multiclass.py weights/mfcc-50.svm.model bof/ 50 labels/test_for_students.csv mfcc-50.svm.csv
```

### MLP classifier

I can train the model through:
```
$ python train_mlp.py bof/ 50 labels/train_val.csv weights/mfcc-50.mlp.model
```
Then, I can get the prediction through:
```
$ python test_mlp.py weights/mfcc-50.mlp.model bof/ 50 labels/test_for_students.csv mfcc-50.mlp.csv
```

## Task 3: Extract SoundNet-Global-Pool

First, I extracted the mp3 file from the videos through:
```
$ for file in videos/*;do filename=$(basename $file .mp4); ffmpeg -y -i $file -ac 1 -ar 22050 -f mp3 mp3/${filename}.mp3; done
```

Then, I extracted features from pool5 layer through:
```
$ python scripts/extract_soundnet_feats.py --feat_layer pool5
```

I trained the MLP Classifier through:
```
$ python train_mlp.py snfp5/ 256 labels/train_val.csv weights/mfcc-50.mlp.model
```

Finally, I predicted the results through:
```
$ python test_mlp.py weights/mfcc-50.mlp.model snfp5/ 256 labels/test_for_students.csv snf.mlp.csv
```


## Task 4: Keep improving your model

I just splited the train and test data in the train_mlp.py file with:
'''
X_train, X_test, y_train, y_test = train_test_split(feat_list, label_list, test_size=0.1, random_state=42)
'''

I used pre-trained PaSST to extract features so I make implement a scripts/extract_PaSST.py file and execute through:
'''
$ python scripts/extract_PaSST.py
'''

I trained the MLP Classifier through:
```
$ python train_mlp.py PaSST/ 1 labels/train_val.csv weights/final.mlp.model
```

Finally, I predicted the results through:
```
$ python test_mlp.py weights/final.mlp.model PaSST/ 1 labels/test_for_students.csv final.mlp.csv
```
### Submission

I make the final predicted submission file through:

$ test_mlp.py weights/final.mlp.model PaSST/ 1 labels/test_for_students.csv final.mlp.csv

Then, I submit final.mlp.csv file to the Kaggle.

( * The final weight/final.mlp.model is different from the file in the code.zip because I tried to improve my model with the other hyperparameters.)

