import numpy as np
from sklearn import metrics, model_selection, multioutput
import lightgbm
from sklearn import ensemble
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from .utils import labels
import pandas as pd

def train_model(x, y, cv_folds, approach_name):


    if approach_name == 'independent':
        # Initialize an empty list to store f1 scores for each fold
        _scores = []

        # Iterate through each fold using its index and the corresponding training/validation indices
        for fold_idx, (idxT, idxV) in enumerate(cv_folds):

            # Define a multioutput classifier model using LGBMClassifier with a random seed of 42
            model = multioutput.MultiOutputClassifier(lightgbm.LGBMClassifier(random_seed=42))

            # Train the model on the training data
            model.fit(x.iloc[idxT], y.iloc[idxT])

            # Generate predictions using the validation data
            p = model.predict(x.iloc[idxV])

            # Calculate the average f1 score for each label using the validation data and the corresponding predictions
            avg_f1_score = np.mean([metrics.f1_score(y.iloc[idxV].iloc[:, i].to_numpy().astype(int), p[:, i].astype(int)) for i in range(4)])

            # Append the average f1 score to the scores list
            _scores.append(avg_f1_score)

            # Print the fold index and the average f1 score
            print("%d\t%.4f" % (fold_idx, avg_f1_score))

        # Return the mean f1 score across all folds
        return np.mean(_scores), model
    


    
    if approach_name == 'classifier_chains':
        _scores = []  # create a list to store F1 scores for each fold

        for fold_idx, (idxT, idxV) in enumerate(cv_folds):  # iterate through each fold
            p_list = []  # create a list to store predictions for each repetition
            avg_f1_score_list = []  # create a list to store F1 scores for each repetition

            for rep in range(3):  # repeat the process 3 times
                # create a Classifier Chain model with LightGBM as the base classifier
                model = multioutput.ClassifierChain(lightgbm.LGBMClassifier(random_seed=42), order="random", cv=5)
                # fit the model on training data
                model.fit(x.iloc[idxT].fillna(0.0), y.iloc[idxT])
                # make predictions on validation data
                p_rep = model.predict(x.iloc[idxV].fillna(0.0))
                # store predictions and F1 score for each repetition
                p_list.append(p_rep)
                avg_f1_score_list.append(np.mean([metrics.f1_score(y.iloc[idxV].iloc[:, i].to_numpy().astype(int), p_rep[:, i].astype(int)) for i in range(4)]))
            
            # calculate average predictions and F1 score across repetitions
            p = np.stack(p_list, axis=0).mean(0)
            avg_f1_score = np.mean([metrics.f1_score(y.iloc[idxV].iloc[:, i].to_numpy().astype(int), p[:, i].astype(int)) for i in range(4)])
            _scores.append(avg_f1_score)
            # print fold index and F1 scores for each repetition and overall F1 score for the fold
            print("%d\t%.4f | %.4f %.4f %.4f" % (fold_idx, avg_f1_score, avg_f1_score_list[0], avg_f1_score_list[1], avg_f1_score_list[2]))
            
        # Return the mean f1 score across all folds
        return np.mean(_scores), model
    


    if approach_name == 'native_extra_trees':

        _scores = [] # initialize a list to store f1 score of each fold

        # loop over each fold of the cross-validation
        for fold_idx, (idxT, idxV) in enumerate(cv_folds):
            
            # initialize an ExtraTreesClassifier model
            model = ensemble.ExtraTreesClassifier()
            
            # fit the model on training data and target labels
            model.fit(x.iloc[idxT].fillna(-100), y.iloc[idxT])
            
            # predict the target labels on validation data
            p = model.predict(x.iloc[idxV].fillna(-100))
            
            # calculate f1 score for each label and take average
            avg_f1_score = np.mean([metrics.f1_score(y.iloc[idxV].iloc[:, i].to_numpy().astype(int), p[:, i].astype(int)) for i in range(4)])
            
            # append the f1 score to the list
            _scores.append(avg_f1_score)
            
        # Return the mean f1 score across all folds
        return np.mean(_scores), model



    if approach_name == 'native_neural_net':

        _scores = []
        for fold_idx, (idxT, idxV) in enumerate(cv_folds):
            norm = keras.layers.Normalization()
            norm.adapt(x.iloc[idxT].fillna(0.0))

            model = keras.Sequential([
                norm,
                keras.layers.Dense(512, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(y.shape[-1])
            ])
            model.compile(
                optimizer=keras.optimizers.Adam(5e-4),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
            )

            x_train = x.iloc[idxT].fillna(0.0).values.astype('float32')
            y_train = y.iloc[idxT].values.astype('int32')
            x_valid = x.iloc[idxV].fillna(0.0).values.astype('float32')
            y_valid = y.iloc[idxV].values.astype('int32')

            hist = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=50, batch_size=32, verbose=0)
            plt.plot(hist.history['loss'], color='tab:blue')
            plt.plot(hist.history['val_loss'], color='tab:red')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
            plt.savefig('training_validation_loss.png')

            p_valid = model.predict(x_valid, verbose=0)
            p_valid = (p_valid > 0.0).astype(int)
            avg_f1_score = np.mean([metrics.f1_score(y.iloc[idxV].iloc[:, i].to_numpy().astype(int), p_valid[:, i].astype(int)) for i in range(4)])
            _scores.append(avg_f1_score)
            print("%d\t%.4f" % (fold_idx, avg_f1_score))
        
        return np.mean(_scores), model
    

    if approach_name == 'multilabel_to_multiclass':
        y_ = y.copy()
        y_['code'] = y.apply(lambda x: '-'.join([str(xi) for xi in x]), axis=1)
        y_codes = y_.drop_duplicates()
        y_codes['idx'] = np.arange(y_codes.shape[0])
        y_class = y_.merge(y_codes, on='code', how='left')['idx']
        print(f"{y_codes.shape[0]} classes")

        _scores = []
        for fold_idx, (idxT, idxV) in enumerate(cv_folds):
            model = lightgbm.LGBMClassifier(random_seed=42)
            model.fit(x.iloc[idxT], y_class.iloc[idxT])

            p = model.predict(x.iloc[idxV])
            p = pd.DataFrame({'idx': p}).merge(y_codes, on='idx', how='left')[labels]
            avg_f1_score = np.mean([metrics.f1_score(y.iloc[idxV].iloc[:, i].to_numpy().astype(int), p.iloc[:, i].astype(int)) for i in range(4)])
            _scores.append(avg_f1_score)
            print("%d\t%.4f" % (fold_idx, avg_f1_score))

        return np.mean(_scores), model



def train_and_evaluate_model(x, y, cv_folds, approach_name):

    print("------------------------")
    print(f"Results for the approach {approach_name}:")
    # Train and evaluate the model using the cross-validation folds
    mean_f1_score, model = train_model(x, y, cv_folds, approach_name)

    # Print the mean f1 score across all folds
    print("--------------")
    print(" \t%.4f" % mean_f1_score)

    return mean_f1_score, model
