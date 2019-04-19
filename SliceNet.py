#%%
#######################################
# Imports and function definitions
#######################################
import os, re
import tensorflow as tf
print(tf.__version__)

import tensorflow_hub as hub
import numpy as np 
from netUtils import batchGen, getTestSet

from keras.layers import Layer, Dense, Input, Lambda, Dropout,\
    Bidirectional, LSTM, Activation, TimeDistributed, Concatenate
from keras import Model, Sequential
from keras.models import model_from_json
import keras.backend as K
from keras.callbacks import ModelCheckpoint

import h5py
import matplotlib.pyplot as plt

print('importing hub')
url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(url)


# Credit: https://github.com/keras-team/keras/issues/3653
def customCatLoss(onehot_labels, logits):
    """Custom categorical loss function incorporating class weights
    Args:
        onehot_labels: onehot encoded labels - shape = [batch x doclength x numclasses]
        logits: logits from predictions from network
    Yields:
        loss: average loss for the mini-batch
    """
    class_weights = [1.0, 10.0, 0.2]
    # computer weights based on onehot labels
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)

    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits)

    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights

    # average to get final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss

class SliceNet():
    def __init__(self, classification=True, pretrain=False, weights_path=None):
        self.pretrain = pretrain # Not implemented
        self.classification = classification
        if self.pretrain:
            self.weights_path = weights_path # Not implemented
            
        self.model = self._defineModel()
    
    def map_docs(self, x):
        """Helper function to embed a give list of sentences using
        the google universal sentence encoder from tfhub"""
        return embed(x, signature='default', as_dict=True)['default']
    
    def UniversalEmbedding(self, x):
        """Embedding layer that calls the map_docs function for each
        document in the batch. The result is an encoding for each sentence
        of each document in the mini-batch"""
        
        # Explicitly cast the input tensor to strings
        x_str = tf.cast(x, tf.string)

        # Embed each sentence of the input tensor
        embedTens = tf.map_fn(self.map_docs, x_str, dtype=tf.float32)
        return embedTens
     
    def _defineModel(self):
        # Define network structure
        encoderIn = Input(shape=[None,], dtype='string', name='encoderIn')
        encoderOut = Lambda(self.UniversalEmbedding, name='encoderOut')(encoderIn)
        lstm1 = Bidirectional(LSTM(256, return_sequences=True), name='lstm_1')(encoderOut)
        lstm2 = Bidirectional(LSTM(256, return_sequences=True), name='lstm_2')(lstm1)
        
        output = Dropout(0.2)(lstm2)
        output = TimeDistributed(Dense(256, activation='relu'))(output)
        output = Dropout(0.2)(output)
        output = TimeDistributed(Dense(64, activation='relu'))(output)
        output = Dropout(0.2)(output)
        
        # Final output is different for classification and regression models
        if self.classification:
            preds = TimeDistributed(Dense(3, activation='softmax'))(output)
            model = Model(inputs=encoderIn, outputs=preds)
            model.compile(loss=customCatLoss, optimizer='adam', metrics=['categorical_accuracy'])
        else:
            preds = TimeDistributed(Dense(1, activation='sigmoid'))(output)
            model = Model(inputs=encoderIn, outputs=preds)
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
        return model

    def train(self, train_files, val_files, batch_size=16, epochs=3, steps_per_epoch=1000, maxlen=None, save=True):
        
        # Define batch generator
        trainGen = batchGen(train_files, batch_size, maxlen, classification=self.classification)
        valGen = batchGen(val_files, 4, maxlen, classification=self.classification)
        
        self.model.summary()
        
        print('Starting Training')
        with tf.Session() as sess:
            K.set_session(sess)
            initOp = [tf.global_variables_initializer(),
                      tf.initializers.tables_initializer()]
            sess.run(initOp)
            
            
            cb = ModelCheckpoint('./models/weights_epoch{epoch:03d}.h5', 
                                         save_weights_only=True, period=2)
            
            history = self.model.fit_generator(trainGen,
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=epochs,
                                          verbose=1,
                                          validation_data=valGen,
                                          validation_steps=10,
                                          callbacks=[cb])
            
            #consider updating loss function from accuracy to precision recall or MSE
            
            if save:
                # Serialize weights to HDF5
                self.model.save_weights('./models/weights_final.h5')
                print("Saved weights to disk")
                
        return history

    def predict(self, test_file, num_samples, weights_path):
        # Get test data and test labels
        X_test, y_test = getTestSet(test_file, num_samples=num_samples)
        
        print('Starting Testing')
        with tf.Session() as sess:
            K.set_session(sess)
            
            initOp = [tf.global_variables_initializer(), tf.initializers.tables_initializer()]
            sess.run(initOp)
            
            # load weights into new model
            self.model.load_weights(weights_path)
            print("Loaded weights from disk")
            
            preds = self.model.predict(X_test)
            
        return preds, y_test
    
    def singlePredict(self, X_test, weights_path):
        
        print('Starting Testing')
        with tf.Session() as sess:
            K.set_session(sess)
            
            initOp = [tf.global_variables_initializer(), tf.initializers.tables_initializer()]
            sess.run(initOp)
            
            # load weights into new model
            self.model.load_weights(weights_path)
            print("Loaded weights from disk")
            
            preds = self.model.predict(X_test)
        
        return preds