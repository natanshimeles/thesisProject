from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K
from batchup import data_source
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plot
import time

#convolutional neural network with 8 layers

class CNN:

    def __init__(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self.trained = False
        print("initializing")

        #first four convolution layer
    def conv_net(self,image):
        if self.trained is False:
            #initialization of convolutional filter
            self.filter1 = tf.Variable(tf.truncated_normal([2, 2, 1, 16],stddev=1e-1), name='weights')#16,2x2 filter
            self.biases1 = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),trainable=True, name='biases')#16 bias

            self.filter2 = tf.Variable(tf.truncated_normal([2,2,16,32],stddev=1e-1),name='weights')#32,2x2 filter
            self.biases2 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),trainable=True, name='biases')#32 bias
            self.filter3 = tf.Variable(tf.truncated_normal([2,2,32,64],stddev=1e-1),name='weights')#64,2x2 filter
            self.biases3 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')#64 bias
            self.filter4 = tf.Variable(tf.truncated_normal([2,2,64,128],stddev=1e-1),name='weights')#128,2x2 filter
            self.biases4 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')#128 bias




        #convolution1
        with tf.name_scope('conv1') as scope:

            conv = self.con2d(image,self.filter1)
            out = self.add_bias(conv, self.biases1)
            self.conv1 = tf.nn.relu(out, name=scope)

        # max pool1
        self.pool1 = self.MaxPooling2D(self.conv1)


        #convolution2
        with tf.name_scope('conv2') as scope:


            conv = self.con2d(self.pool1,self.filter2)
            out = self.add_bias(conv, self.biases2)
            self.conv2 = tf.nn.relu(out, name=scope)

        #max  pool2
        self.pool2 = self.MaxPooling2D(self.conv2)

        #convolution3
        with tf.name_scope('conv3') as scope:


            conv = self.con2d(self.pool2,self.filter3)
            out =self.add_bias(conv, self.biases3)
            self.conv3 = tf.nn.relu(out, name=scope)


        #max pool3
        self.pool3 = self.MaxPooling2D(self.conv3)

        #convolution4
        with tf.name_scope('conv4') as scope:


            conv = self.con2d(self.pool3,self.filter4)
            out =self.add_bias(conv, self.biases4)

        #max pool4
        self.pool4 = self.MaxPooling2D(out)



        return  self.pool4



    #four feed forward layers
    def feed_net(self):
        #make input ready for the next layer
        self.flatten_layer= tf.layers.Flatten()(self.pool4)
        shape = int(np.prod(self.pool4.get_shape()[1:]))

        if self.trained is False:
            #initialization
            self.fc1w = tf.Variable(tf.truncated_normal([shape, 1024],stddev=1e-1), name='weights')#first layer with 1024 neurons
            self.fc1b = tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32),trainable=True, name='biases')#bias for first kayer
            self.fc2w = tf.Variable(tf.truncated_normal([1024, 512],stddev=1e-1), name='weights')# second layers with 512 neurons
            self.fc2b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32),trainable=True, name='biases')#bias for second layer
            self.fc3w = tf.Variable(tf.truncated_normal([512, 256],stddev=1e-1), name='weights')#third layer with 256 neurons
            self.fc3b = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')#bias for third layer
            self.fc4w = tf.Variable(tf.truncated_normal([256, 33],stddev=1e-1), name='weights')#output layer with 33 neurons
            self.fc4b = tf.Variable(tf.constant(1.0, shape=[33], dtype=tf.float32),trainable=True, name='biases')#bias for last layer
            self.trained = True
            print("variables initialized ")
        else:
            print("using initialized values")

        #FEED FORWARD 1
        with tf.name_scope('fc_1') as scope:
            fc1l = self.add_bias(tf.matmul(self.flatten_layer, self.fc1w), self.fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.fc1 = self.dropout(self.fc1)
        #FEED FORWARD 2
        with tf.name_scope('fc_2') as scope:


            fc2l = self.add_bias(tf.matmul(self.fc1, self.fc2w), self.fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.fc2 = self.dropout(self.fc2)

        #FEED FORWARD 3
        with tf.name_scope('fc_3') as scope:

            fc3l = self.add_bias(tf.matmul(self.fc2, self.fc3w), self.fc3b)
            self.fc2 = self.dropout(self.fc2)
            self.fc3=tf.nn.relu(fc3l)
        #FEED FORWARD 4
        with tf.name_scope('fc_4') as scope:

            fc4l = self.add_bias(tf.matmul(self.fc3, self.fc4w), self.fc4b)
            self.logits=fc4l


    def train(self,image,label,epoches=100):
        #IMAGE PLACE HOLDER
        self.image =  tf.placeholder(dtype=tf.float32, shape=[None, 32,32,1])
        #LABEL PLACE HOLDER
        l = tf.placeholder(dtype=tf.int64, shape=[143])

        #FORWARD PROPAGATION
        self.conv_net(self.image)
        self.feed_net()
        predicted_res = self.logits

        l_r = tf.placeholder(tf.float32)

        finaloutput = tf.nn.softmax(self.logits, name="softmax")

        prediction_labels = tf.argmax(finaloutput, axis=1, name="output")



        correct_prediction = tf.equal(tf.cast(prediction_labels, tf.int64), tf.cast(l, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))



        batch_size = 143
        #INITIALIZE LOSS
        loss =  tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=l, logits=predicted_res))
        #INITALIZE OPTIMIZER
        optimizer = tf.train.AdamOptimizer(learning_rate = l_r)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())




        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for epoch in range(epoches):#TRAINING
                epoch_loss = 0
                i = 0
                epoch_accu = 0

                while i < len(image):

                    start = i
                    end = i + batch_size
                    i =  start + batch_size
                    batch_x = np.array(image[start:end])
                    l_ = np.array(label[start:end])

                    loss_val , _,acc,pre= sess.run([loss,train_op,accuracy,correct_times_in_batch],feed_dict={self.image:batch_x , l:l_ , self.keep_prob:0.5,l_r:0.001})

                    epoch_loss = epoch_loss + loss_val
                    epoch_accu = epoch_accu + pre

                print("epcoh ",epoch+1," loss: ",epoch_loss," Acc: ",epoch_accu/len(image))
                self.save_network(sess)#SAVE THE VALUES AFTER EACH TRAINING
            total_parameters = 0
        print("train")


    def evaluate(self,image,label,comments,batch = 161):#TEST THE MODEL WITH ACCURACY
        saver = tf.train.Saver()
        self.image =  tf.placeholder(dtype=tf.float32, shape=[None, 32,32,1])
        self.conv_net(self.image)
        self.feed_net()
        l = tf.placeholder(dtype=tf.int64, shape=[batch])
        finaloutput = tf.nn.softmax(self.logits, name="softmax")
        prediction_labels = tf.argmax(finaloutput, axis=1, name="output")
        correct_prediction = tf.equal(tf.cast(prediction_labels, tf.int64), tf.cast(l, tf.int64))#GET THE NUMBER OF CORRECT PREDICTED VALUES

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
        with tf.Session() as sess:
            total_accu = 0
            i = 0
            sess.run([tf.initialize_all_variables()])
            saver.restore(sess, "new_models/model.ckpt")#LOAD PARAMATERS
            while i < len(image):
                start = i
                end = i + batch
                i =  start + batch
                batch_images = images[start:end]
                batch_labels = labels[start:end]
                feed_dict={self.image:batch_images,l:batch_labels,self.keep_prob:1.0}
                correct_times = sess.run(correct_times_in_batch,feed_dict)
                total_accu = total_accu + correct_times
            print(comments," Accuracy: ",total_accu/len(image))




    def save_network(self,sess):#SAVE THE parameters
            saver = tf.train.Saver()
            save_path = saver.save(sess, "new_models/model.ckpt")
    def load_network(self,sess):#LOAD PARAMETERS
        saver = tf.train.Saver()
        saver.saver.restore(sess, "new_models/model.ckpt")
    def predict(self,image):#PREDICT OUTPUT
        self.image = tf.placeholder(dtype=tf.float32, shape=[None, 32,32,1])
        self.conv_net(self.image)
        self.feed_net()
        self.predictions = tf.nn.softmax(self.logits)
        predictions =  tf.argmax(self.predictions,1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run([tf.initialize_all_variables()])
            saver.restore(sess, "new_models/model.ckpt")
            feed_dict={self.image:image,self.keep_prob:1.0}
            predictions = sess.run([predictions],feed_dict)

        return predictions

    def con2d(self,images,filter):
        ##                                stride= move 1 pixle in all sides
        return tf.nn.conv2d(images,filter,[1,1,1,1],padding="SAME",use_cudnn_on_gpu=True)
    def MaxPooling2D(self,images):
            #                        size of window  same as   movment       no image shrinking (padding can be same or valid)
        return tf.nn.max_pool(images,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    def add_bias(self,value,bias):
        return tf.nn.bias_add(value,bias)
    def dropout(self,input):
        return tf.nn.dropout(input, keep_prob=self.keep_prob)

def divide_data(image,labels):#SEGMENT DATA TO TRAINING, TESTING AND CROSS validation SET
    images,labels = shuffle(image,labels)
    image_train, image_test, label_train, label_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    image_developdemt_set, image_test, label_developdemt_set, label_test = train_test_split(image_test, label_test, test_size=0.5, random_state=42)
    return image_train, image_developdemt_set, image_test, label_train,label_developdemt_set,label_test

if __name__ == '__main__':#MAIN BODY
    cnn = CNN()#INITALIZE
    fulldata = np.load("data/fd7.npz")#LOAD DATA
    images =fulldata["arr_0"]
    images = np.reshape(images,[-1,32,32,1])
    images =images.astype(float)
    labels =fulldata["arr_1"]
    image_train, image_developdemt_set, image_test, label_train,label_developdemt_set,label_test = divide_data(images,labels)#SEGMENT DATA

    start_time = time.time()#INITALIZE STARTING TIME
    cnn.train(image_train, label_train)
    end_time = time.time()
    total_time = end_time-start_time

    print("total time:",total_time,"s")
    cnn.evaluate(image_test,label_test,"test on testing set")
    cnn.evaluate(image_developdemt_set,label_developdemt_set,"test on cross validation")
