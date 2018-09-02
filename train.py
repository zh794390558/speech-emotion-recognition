#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:05:03 2018

@author: hxj
"""

import numpy as np
import tensorflow as tf
import crnn
import pickle
import os
from utils import dense_to_one_hot
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion

FLAGS = crnn.FLAGS

def load_data():
    f = open(FLAGS.data_path, 'rb')
    train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label,\
        Test_label, pernums_test, pernums_valid = pickle.load(f)
    return train_data,train_label, valid_data, Valid_label

def train(train_dir=None, model_dir=None, mode='train'):
    model = crnn.CRNN(mode)
    model._build_model()
    global_step = tf.Variable(0, trainable=False)
    #sess1 = tf.InteractiveSession()

    #load training data
    train_data, train_label, valid_data, Valid_label = load_data()
    train_label = dense_to_one_hot(train_label, 4)
    Valid_label = dense_to_one_hot(Valid_label, 4)
    training_size = train_data.shape[0]

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = model.labels, logits = model.logits)
        loss = tf.reduce_mean(cross_entropy)
#        print model.logits.get_shape()  
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(model.softmax, 1), tf.argmax(model.labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.momentum, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.name_scope("train_step"):
        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                        global_step,
                                        training_size/FLAGS.train_batch_size,
                                        FLAGS.decay_rate,
                                        staircase=True)
        #print (lr.eval())        
        train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)
        for e in range(FLAGS.num_epochs):
            print(type(train_label))
            print(train_label.shape)
            index = np.arange(training_size)
            np.random.shuffle(index)
            train_data = train_data[index]
            train_label = train_label[index]

            for i in range(int(training_size / FLAGS.train_batch_size) + 1):
                start = (i * FLAGS.train_batch_size) % training_size
                end = min(start + FLAGS.train_batch_size, training_size)
                _, loss_value, step, acc, softmax = sess.run(
                    [train_op, loss, global_step, accuracy, model.softmax],
                    feed_dict={
                        model.inputs: train_data[start:end],
                        model.labels: train_label[start:end]
                    }
                )

                if i % 10 == 0:
                    print("After epoch:%d, step: %d, loss on training batch is %.2f, accuracy is %.3f." %(e, step, loss_value, acc))
                    saver.save(
                            sess, 
                            os.path.join(FLAGS.checkpoint, FLAGS.model_name), 
                            global_step = global_step)
                    train_acc_uw = recall(np.argmax(softmax, 1), np.argmax(train_label[start:end], 1), average='macro')
                    train_acc_w = recall(np.argmax(softmax, 1), np.argmax(train_label[start:end], 1), average='weighted')
                    train_conf = confusion(np.argmax(softmax, 1), np.argmax(train_label[start:end], 1))
                    print("train_UA: %3.4g" % train_acc_uw) 
                    print("train_WA: %3.4g" % train_acc_w)
                    print('Confusion Matrix:["ang","sad","hap","neu"]')
                    print(train_conf)

                if i % 20 == 0:
                    #for validation data
                    valid_size = len(valid_data)
                    valid_iter = divmod((valid_size), FLAGS.valid_batch_size)[0]
                    y_pred_valid = np.empty((valid_size, 4),dtype=np.float32)
                    index = 0
                    cost_valid = 0
                    if(valid_size < FLAGS.valid_batch_size):
                        validate_feed = { 
                            model.inputs:valid_data, 
                            model.labels:Valid_label
                        }
                        y_pred_valid, p_loss = sess.run([model.softmax, cross_entropy], feed_dict = validate_feed)
                        cost_valid = cost_valid + np.sum(p_loss)
                    else:
                        print(valid_data.shape)
                        for v in range(valid_iter):
                            v_begin = v * FLAGS.valid_batch_size
                            v_end = (v+1) * FLAGS.valid_batch_size
                            if(v == valid_iter-1):
                                if(v_end < valid_size):
                                    v_end = valid_size
                            validate_feed = { 
                                model.inputs:valid_data[v_begin:v_end], 
                                model.labels:Valid_label[v_begin:v_end]
                            }
                            p_loss, y_pred_valid[v_begin:v_end,:] = sess.run([cross_entropy, model.softmax], feed_dict = validate_feed)
                            cost_valid = cost_valid + np.sum(p_loss)
                    cost_valid = cost_valid/valid_size
                    print(np.argmax(y_pred_valid, 1))
                    print(np.argmax(Valid_label, 1))
                    valid_acc_uw = recall(np.argmax(Valid_label,1), np.argmax(y_pred_valid,1), average='macro')
                    valid_acc_w = recall(np.argmax(Valid_label, 1), np.argmax(y_pred_valid,1), average='weighted')
                    valid_conf = confusion(np.argmax(Valid_label, 1),np.argmax(y_pred_valid,1))

                    print('----------segment metrics---------------') 
                    print("valid_UA: %3.4g" % valid_acc_uw) 
                    print("valid_WA: %3.4g" % valid_acc_w)
                    print('Valid Confusion Matrix:["ang","sad","hap","neu"]')
                    print(valid_conf)
                    print('----------segment metrics---------------') 


if __name__=='__main__':
    train()
