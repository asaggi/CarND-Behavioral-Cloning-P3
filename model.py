import pandas as pd
import numpy as np
import argparse, os

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from preprocess import batchGenerator, ISHAPE, getBool
from sklearn.model_selection import train_test_split

np.random.seed(0)

def buildModel(args):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=ISHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keepProb))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def trainModel(model, args, Xtrain, Xvalid, ytrain, yvalid):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learningRate))

    model.fit_generator(batchGenerator(args.dataDir, Xtrain, ytrain, args.batchSize, True),
                        args.samplesPerEpoch,
                        args.nbEpoch,
                        max_q_size=1,
                        validation_data=batchGenerator(args.dataDir, Xvalid, yvalid, args.batchSize, False),
                        nb_val_samples=len(Xvalid),
                        callbacks=[checkpoint],
                        verbose=1)
### MAIN ###
def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-t', help='Test Size Fraction',        dest='testSize',         type=float,     default=0.2)
    parser.add_argument('-l', help='Learning Rate Defined',     dest='learningRate',     type=float,     default=1.0e-4)
    parser.add_argument('-d', help='Data Directory',            dest='dataDir',          type=str,       default='data')
    parser.add_argument('-o', help='Save best Only',            dest='save_best_only',   type=getBool,   default='true')
    parser.add_argument('-b', help='Batch Size',                dest='batchSize',        type=int,       default=40)
    parser.add_argument('-n', help='# Epochs',                  dest='nbEpoch',          type=int,       default=10)
    parser.add_argument('-s', help='Samples/Epoch',             dest='samplesPerEpoch',  type=int,       default=20000)
    parser.add_argument('-k', help='Drop Out',                  dest='keepProb',         type=float,     default=0.5)

    args = parser.parse_args()

    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))

    data = loadData(args)
    model = buildModel(args)
    trainModel(model, args, *data)

    
def loadData(args):
    dataDf = pd.read_csv(os.path.join(args.dataDir, 'driving_log.csv'))

    X = dataDf[['center', 'left', 'right']].values
    y = dataDf['steering'].values

    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=args.testSize, random_state=0)

    return Xtrain, Xvalid, ytrain, yvalid


 ##### START HERE #####   
if __name__ == '__main__':
    main()


