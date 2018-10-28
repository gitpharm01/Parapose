import tensorflow as tf
import numpy as np
import time
from imageLoader import getPaddedROI,training_data_feeder
import math
'''
created by Cid Zhang 
a sub-model for human pose estimation
'''

#function to locate the point with maximum value on similarity map
def max_sim_point( sim_map , orig_feature_map_size, image_input_size ):
    
    #get the position of  max value
    max_pos = tf.argmax( sim_map , output_type=tf.int32)
    
    max_value = sim_map[max_pos]
    #reshape the similarity map to 2d format
    sim_map= tf.reshape(sim_map,[orig_feature_map_size , orig_feature_map_size])
    p = tf.where (tf.equal (sim_map,max_value ) ) 

    def cond_t(p):
        return p[0]
    def cond_f(p):
        return p
    joint = tf.cond(tf.shape(p)[0] > 1 , lambda:cond_t(p) , lambda:cond_f(p) ) 
    x = tf.cast(joint[0], tf.float32) * tf.cast((image_input_size/orig_feature_map_size),tf.float32)
    y = tf.cast(joint[1], tf.float32) * tf.cast((image_input_size/orig_feature_map_size),tf.float32)
    return ( x , y )

def truncated_normal_var(name,shape,dtype):
    return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.01)))
def zero_var(name,shape,dtype):
    return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

roi_size = 23
image_input_size = 301

#input placeholders
#batch1 hints
inputs_b1h1 = tf.placeholder(tf.float32, ( 16, roi_size, roi_size, 3), name='inputs_b1h1')
inputs_b1h2 = tf.placeholder(tf.float32, ( 16, roi_size, roi_size, 3), name='inputs_b1h2')


inputs_s = tf.placeholder(tf.float32, (None, image_input_size, image_input_size, 3), name='inputs_s')
labels = tf.placeholder(tf.float32,(16,76,76), name='labels')

#define the model
def paraNet(input):
    out_l1 = tf.layers.conv2d(input, 8, [3, 3],strides=(2, 2), padding ='valid' ,name='para_conv_1')
    out_l1 = tf.nn.relu6(out_l1)
    out_l2 = tf.layers.conv2d(out_l1, 16, [3, 3],strides=(2, 2), padding ='valid' ,name='para_conv_2')
    out_l2 = tf.nn.relu6(out_l2)
    out_l3 = tf.layers.conv2d(out_l2, 32, [5, 5],strides=(1, 1), padding ='valid' ,name='para_conv_3')
    return out_l3

#network pipeline to create the first Hint Hash Sets (Three batches)
with tf.variable_scope('conv'):
    out_b1h1_l3 = paraNet(inputs_b1h1)
    #flatten and binerize the hashs
    out_b1h1_l3 =tf.squeeze(  tf.round(tf.nn.sigmoid(out_b1h1_l3)) )



#network pipeline to create the Second Hint Hash Sets
with tf.variable_scope('conv', reuse=True):
    out_b1h2_l3 = paraNet(inputs_b1h2)
    #flatten and binerize the hashs
    out_b1h2_l3 =tf.squeeze( tf.round(tf.nn.sigmoid(out_b1h2_l3)) )



with tf.variable_scope('conv', reuse=True):
    out_2_l1 = tf.layers.conv2d(inputs_s,  8, [3, 3],strides=(2, 2), padding ='same' ,name='para_conv_1')
    out_2_l1 = tf.nn.relu6(out_2_l1)
    out_2_l2 = tf.layers.conv2d(out_2_l1, 16, [3, 3],strides=(2, 2), padding ='same' ,name='para_conv_2')
    out_2_l2 = tf.nn.relu6(out_2_l2)
    out_2_l3 = tf.layers.conv2d(out_2_l2, 32, [5, 5],strides=(1, 1), padding ='same' ,name='para_conv_3')
    #binerize the value
    out_2_l3 = tf.round(tf.nn.sigmoid(out_2_l3))
    #iterate through each pixel of the final feature map 
    #turn the value into hashes 
    orig_feature_map_size = tf.shape(out_2_l3)[1]
    
    #calculate Hamming distance maps
    map0 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[0] )) , axis=3 ), 
                       tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[0] )) , axis=3 )  ) 
    map1 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[1] )) , axis=3 ), 
                       tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[1] )) , axis=3 )  ) 
    map2 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[2] )) , axis=3 ), 
                       tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[2] )) , axis=3 )  ) 
    map3 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[3] )) , axis=3 ), 
                       tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[3] )) , axis=3 )  ) 
    map4 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[4] )) , axis=3 ), 
                       tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[4] )) , axis=3 )  ) 
    map5 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[5] )) , axis=3 ), 
                       tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[5] )) , axis=3 )  ) 
    map6 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[6] )) , axis=3 ), 
                       tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[6] )) , axis=3 )  ) 
    map7 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[7] )) , axis=3 ), 
                       tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[7] )) , axis=3 )  ) 
    map8 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[8] )) , axis=3 ), 
                       tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[8] )) , axis=3 )  ) 
    map9 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[9] )) , axis=3 ), 
                       tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[9] )) , axis=3 )  ) 
    map10 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[10] )) , axis=3 ), 
                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[10] )) , axis=3 )  ) 
    map11 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[11] )) , axis=3 ), 
                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[11] )) , axis=3 )  ) 
    map12 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[12] )) , axis=3 ), 
                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[12] )) , axis=3 )  ) 
    map13 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[13] )) , axis=3 ), 
                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[13] )) , axis=3 )  ) 
    map14 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[14] )) , axis=3 ), 
                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[14] )) , axis=3 )  ) 
    map15 = tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h2_l3[15] )) , axis=3 ), 
                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_b1h1_l3[15] )) , axis=3 )  ) 
    totoal_map =tf.div( tf.concat([map0, map1, map2, map3, map4, map5, map6, map7,
                                   map8, map9, map10,map11,map12, map13, map14, map15], 0) , 32)
    loss = tf.nn.l2_loss(labels- totoal_map  , name = 'loss'  )

#ValueError: No gradients provided for any variable, check your graph for ops that do not support gradients, between variables 
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss )


init =  tf.global_variables_initializer()
batchsize = 3

with tf.Session() as sess:
    #writer = tf.summary.FileWriter("./variable_graph",graph = sess.graph)
    sess.run(init)
    
    #load image from dataset(train set)
    joint_data_path = "./custom_data.json"
    train_val_path = "./train_val_indices.json"
    imgpath = "./000/"
    input_size = 301
    hint_roi_size = 23
    
    hintSet01_norm_batch = []
    hintSet02_norm_batch = []
    t_img_batch = []
    t_label_norm_batch = []
    #load data
    hintSet01,hintSet02,t_img,t_label_norm = training_data_feeder(joint_data_path, train_val_path, imgpath, input_size, hint_roi_size )
    #Normalize the image pixel values to 0~1
    hintSet01_norm = []
    hintSet02_norm = []

    t_img = np.float32(t_img /255.0)
    #print(type(t_label_norm))
    for rois in hintSet01:
        tmp = np.float32(rois / 255.0)
        hintSet01_norm.append(tmp.tolist())
    for rois in hintSet02:
        tmp = np.float32(rois / 255.0)
        hintSet02_norm.append(tmp.tolist())
    
    print(tf.trainable_variables())
    #print(hash_set01)
    #print(out_2_l3)
    temp = sess.run(totoal_map , feed_dict={inputs_s:  [t_img]  , 
                                        inputs_b1h1: hintSet01_norm, inputs_b1h2: hintSet02_norm, #batch no.1
                                        labels: t_label_norm 
                                                       })
    print(temp)
    print(np.shape(temp))


    
'''   
    for i in range(1000):
        hintSet01_norm, hintSet02_norm, t_img, t_label_norm = get_train_data(batchsize)
        sess.run(train_step, feed_dict={inputs_s: [ t_img ] , 
                                        inputs_b1h1: hintSet01_norm[0], inputs_b1h2: hintSet02_norm[0], #batch no.1
                                        inputs_b2h1: hintSet01_norm[1], inputs_b2h2: hintSet02_norm[1], #batch no.2
                                        inputs_b3h1: hintSet01_norm[2], inputs_b3h2: hintSet02_norm[2], #batch no.3
                                        labels: t_label_norm 
                                                        })
        if i % 50 == 0:
            print(total_loss)
    
'''
