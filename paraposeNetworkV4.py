import tensorflow as tf
import numpy as np
import time
from imageLoader import getPaddedROI,training_data_feeder
import math
import cv2
'''
created by Cid Zhang 
a sub-model for human pose estimation
'''
tf.reset_default_graph()

def truncated_normal_var(name,shape,dtype):
    return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.01)))
def zero_var(name,shape,dtype):
    return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

roi_size = 23
image_input_size = 301

#input placeholders
#batch1 hints
inputs_b1h1 = tf.placeholder(tf.float32, ( 16, roi_size, roi_size, 3), name='inputs_b1h1')
#inputs_b1h2 = tf.placeholder(tf.float32, ( 16, roi_size, roi_size, 3), name='inputs_b1h2')


inputs_s = tf.placeholder(tf.float32, (None, image_input_size, image_input_size, 3), name='inputs_s')
labels = tf.placeholder(tf.float32,(16,76,76), name='labels')

#define the model

def paraNet(inputs, inputs_s , ground_truth_labels ):
    with tf.variable_scope('conv'):
        out_l1 = tf.layers.conv2d(inputs, 16, [3, 3],strides=(2, 2), padding ='valid' ,name='para_conv_1')
        out_l1r = tf.nn.relu(out_l1)
        out_l2 = tf.layers.conv2d(out_l1r, 48, [3, 3],strides=(2, 2), padding ='valid' ,name='para_conv_2')
        out_l2r = tf.nn.relu(out_l2)
        out_l3 = tf.layers.conv2d(out_l2r, 96, [5, 5],strides=(1, 1), padding ='valid' ,name='para_conv_3')
        out_l3r = tf.nn.relu(out_l3)
        out_l4 = tf.layers.conv2d(out_l3r, 32, [1, 1],strides=(1, 1), padding ='valid' ,name='para_conv_4')
        hint = tf.squeeze(  tf.sign( tf.sigmoid(out_l4) ) )

    with tf.variable_scope('conv', reuse=tf.AUTO_REUSE ):
        out_2_l1 = tf.layers.conv2d(inputs_s,  16, [3, 3],strides=(2, 2), padding ='same' ,name='para_conv_1')
        out_2_l1r = tf.nn.relu(out_2_l1)
        out_2_l2 = tf.layers.conv2d(out_2_l1r, 48, [3, 3],strides=(2, 2), padding ='same' ,name='para_conv_2')
        out_2_l2r = tf.nn.relu(out_2_l2)
        out_2_l3 = tf.layers.conv2d(out_2_l2r, 96, [5, 5],strides=(1, 1), padding ='same' ,name='para_conv_3')
        out_2_l3r = tf.nn.relu(out_2_l3)
        out_2_l4 = tf.layers.conv2d(out_2_l3r, 32, [1, 1],strides=(1, 1), padding ='same' ,name='para_conv_4')
        sample =tf.sign( tf.sigmoid(out_2_l4))
    
    map0 = tf.reduce_sum ( tf.abs (tf.subtract( hint[0] , sample ) ) , axis=3 )  
    map1 = tf.reduce_sum ( tf.abs (tf.subtract( hint[1] , sample ) ) , axis=3 )  
    map2 = tf.reduce_sum ( tf.abs (tf.subtract( hint[2] , sample ) ) , axis=3 )  
    map3 = tf.reduce_sum ( tf.abs (tf.subtract( hint[3] , sample ) ) , axis=3 )  
    map4 = tf.reduce_sum ( tf.abs (tf.subtract( hint[4] , sample ) ) , axis=3 )  
    map5 = tf.reduce_sum ( tf.abs (tf.subtract( hint[5] , sample ) ) , axis=3 )  
    map6 = tf.reduce_sum ( tf.abs (tf.subtract( hint[6] , sample ) ) , axis=3 )  
    map7 = tf.reduce_sum ( tf.abs (tf.subtract( hint[7] , sample ) ) , axis=3 )  
    map8 = tf.reduce_sum ( tf.abs (tf.subtract( hint[8] , sample ) ) , axis=3 )  
    map9 = tf.reduce_sum ( tf.abs (tf.subtract( hint[9] , sample ) ) , axis=3 )  
    map10 = tf.reduce_sum ( tf.abs (tf.subtract( hint[10] , sample ) ) , axis=3 )  
    map11 = tf.reduce_sum ( tf.abs (tf.subtract( hint[11] , sample ) ) , axis=3 )  
    map12 = tf.reduce_sum ( tf.abs (tf.subtract( hint[12] , sample ) ) , axis=3 )  
    map13 = tf.reduce_sum ( tf.abs (tf.subtract( hint[13] , sample ) ) , axis=3 )  
    map14 = tf.reduce_sum ( tf.abs (tf.subtract( hint[14] , sample ) ) , axis=3 )  
    map15 = tf.reduce_sum ( tf.abs (tf.subtract( hint[15] , sample ) ) , axis=3 )  
    
    totoal_map =tf.div( tf.concat([map0, map1, map2, map3, map4, map5, map6, map7,
                                   map8, map9, map10,map11,map12, map13, map14, map15], 0) , 64)
    loss = tf.nn.l2_loss( totoal_map -  ground_truth_labels , name = 'loss'  )

    return loss, totoal_map

loss, totoal_map = paraNet(inputs_b1h1, inputs_s, labels)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init =  tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    #writer = tf.summary.FileWriter("./variable_graph",graph = sess.graph)
    sess.run(init)

    #load image from dataset(train set)
    joint_data_path = "./custom_data.json"
    train_val_path = "./train_val_indices.json"
    imgpath = "./000/"
    input_size = 301
    hint_roi_size = 23
    '''
    #load data
    hintSet01,hintSet02,t_img,t_label_norm = training_data_feeder(joint_data_path, train_val_path, imgpath, input_size, hint_roi_size )
    #Normalize the image pixel values to 0~1
    hintSet01_norm = []
    hintSet02_norm = []

    t_img =[ np.float32(t_img /255.0) ]
    #print(type(t_img))
    #print(np.shape(t_img))
    #print(type(t_label_norm))
    for rois in hintSet01:
        tmp = np.float32(rois / 255.0)
        hintSet01_norm.append(tmp.tolist())
    for rois in hintSet02:
        tmp = np.float32(rois / 255.0)
        hintSet02_norm.append(tmp.tolist())
    
    loss_value , total_map_value = sess.run ([loss, totoal_map], feed_dict = {inputs_s:  t_img, 
                                                                              inputs_b1h1: hintSet01_norm, 
                                                                              labels: t_label_norm
                                                                              })
    print("-----loss value:",loss_value)
    print("-----total_map_value:", total_map_value[0,0] )
    print("-----label_value", t_label_norm[0,0] )
    #cv2.imshow("t_img",t_img[0])
    #for img in t_label_norm:
    #    print(img)
    #    cv2.imshow("hint", img)
    #    cv2.waitKey(0)

    #print(tf.trainable_variables())
    #print(hash_set01)
    #print(out_2_l3)
    '''
    #saver.restore(sess, "./temp_model/model5.ckpt")


    for i in range(5000):

        #load data
        hintSet01,hintSet02,t_img,t_label_norm = training_data_feeder(joint_data_path, train_val_path, imgpath, input_size, hint_roi_size )
        #Normalize the image pixel values to 0~1
        hintSet01_norm = []
        hintSet02_norm = []

        t_img =[ np.float32(t_img /255.0) ]
        #print(type(t_img))
        #print(np.shape(t_img))
        #print(type(t_label_norm))
        for rois in hintSet01:
            tmp = np.float32(rois / 255.0)
            hintSet01_norm.append(tmp.tolist())
        for rois in hintSet02:
            tmp = np.float32(rois / 255.0)
            hintSet02_norm.append(tmp.tolist())
        loss_val, _ = sess.run([loss, train_step] , 
                          feed_dict = {inputs_s:  t_img, 
                                       inputs_b1h1: hintSet01_norm, 
                                       labels: t_label_norm })
        if i % 50 == 0:
            print(loss_val)
    
    #save_path = saver.save(sess, "./temp_model/model" + '5' + ".ckpt")
    #print(temp)
    #print(np.shape(temp))

