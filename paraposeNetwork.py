import tensorflow as tf
import numpy as np
import time
from imageLoader import getPaddedROI,training_data_feeder
import math
'''
created by Cid Zhang 
a sub-model for human pose estimation
'''
#input data feeder !!! important !!! The hintSetx_norm_batches are 5d tensors!!! To accommodate, the batch size are fixed to 2
def get_train_data(batch_size):
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
    for i in range(batch_size):
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
 
        hintSet01_norm_batch.append(hintSet01_norm)
        hintSet02_norm_batch.append(hintSet02_norm)

        t_img_batch.append(t_img)
        t_label_norm_batch.append( t_label_norm)

    return hintSet01_norm_batch, hintSet02_norm_batch, t_img_batch, t_label_norm_batch

# locate minimum value Position( the shortest Hamming distance)
def locateMin_and_get_loss( distance_map ,original_map_size, label  ):
    #locate the minimum value position
    tmin = tf.argmin( tf.reshape(distance_map,[-1]), output_type=tf.int32)
    #!!!!!must notice!!!! The divisor to normalize the final position was set manually (now as 76)
    #It was because I cannot get the desired result with "original_map_size"
    pos = tf.cast( ((tf.floormod(tmin, original_map_size) +1 ) / 76 , (tf.floordiv(tmin , original_map_size) +1 )/ 76),tf.float32)
    
    dist =tf.abs( tf.norm(label - pos, ord='euclidean'))
    return dist,pos

def get_total_loss_and_result( out_2_l3 ,  out_h2_l3 , out_h1_l3 , orig_feature_map_size , labels ):
    dist0,pos0 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[0] )) , axis=2 ), 
                                                      tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[0] )) , axis=2 )  ) , orig_feature_map_size, labels[0])
    dist1,pos1 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[1] )) , axis=2 ), 
                                                      tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[1] )) , axis=2 )  ) , orig_feature_map_size, labels[1])
    dist2,pos2 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[2] )) , axis=2 ), 
                                                      tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[2] )) , axis=2 )  ) , orig_feature_map_size, labels[2])
    dist3,pos3 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[3] )) , axis=2 ), 
                                                      tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[3] )) , axis=2 )  ) , orig_feature_map_size, labels[3])
    dist4,pos4 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[4] )) , axis=2 ), 
                                                      tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[4] )) , axis=2 )  ) , orig_feature_map_size, labels[4])
    dist5,pos5 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[5] )) , axis=2 ), 
                                                      tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[5] )) , axis=2 )  ) , orig_feature_map_size, labels[5])
    dist6,pos6 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[6] )) , axis=2 ), 
                                                      tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[6] )) , axis=2 )  ) , orig_feature_map_size, labels[6])
    dist7,pos7 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[7] )) , axis=2 ), 
                                                      tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[7] )) , axis=2 )  ) , orig_feature_map_size, labels[7])
    dist8,pos8 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[8] )) , axis=2 ), 
                                                      tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[8] )) , axis=2 )  ) , orig_feature_map_size, labels[8])
    dist9,pos9 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[9] )) , axis=2 ), 
                                                      tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[9] )) , axis=2 )  ) , orig_feature_map_size, labels[9])
    dist10,pos10 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[10])) , axis=2 ), 
                                                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[10])) , axis=2 )  ) , orig_feature_map_size, labels[10])
    dist11,pos11 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[11])) , axis=2 ), 
                                                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[11])) , axis=2 )  ) , orig_feature_map_size, labels[11])
    dist12,pos12 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[12])) , axis=2 ), 
                                                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[12])) , axis=2 )  ) , orig_feature_map_size, labels[12])
    dist13,pos13 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[13])) , axis=2 ), 
                                                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[13])) , axis=2 )  ) , orig_feature_map_size, labels[13])
    dist14,pos14 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[14])) , axis=2 ), 
                                                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[14])) , axis=2 )  ) , orig_feature_map_size, labels[14])
    dist15,pos15 = locateMin_and_get_loss(  tf.minimum( tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h2_l3[15])) , axis=2 ), 
                                                        tf.reduce_sum (tf.abs(tf.subtract( out_2_l3, out_h1_l3[15])) , axis=2 )  ) , orig_feature_map_size, labels[15])
    #total_loss =tf.add_n( tf.add_n( tf.add_n( tf.add_n( tf.add_n( tf.add_n( tf.add_n( tf.add_n( tf.add_n( tf.add_n( tf.add_n( tf.add_n( (tf.add_n( tf.add_n( tf.add_n(dist0 , dist1) , dist2) , dist3) , dist4 ), dist5 ), dist6 ), dist7 ), dist8) , dist9) , dist10) , dist11) , dist12 ), dist13 ), dist14 ), dist15)
    total_loss = tf.stack([dist0 , dist1 , dist2 , dist3 , dist4 , dist5 , dist6 , dist7 ,
                           dist8 , dist9 , dist10 , dist11 , dist12 , dist13 , dist14 , dist15], axis=0)
    total_loss = tf.reduce_sum( total_loss )
    final_output = tf.stack([pos0,  pos1, pos2,  pos3,  pos4,  pos5,  pos6,  pos7
                    ,pos8 ,pos9 ,pos10 ,pos11 ,pos12 ,pos13 ,pos14,pos15], axis = 0) 
    return total_loss, final_output


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
#batch2 hints
inputs_b2h1 = tf.placeholder(tf.float32, ( 16, roi_size, roi_size, 3), name='inputs_b2h1')
inputs_b2h2 = tf.placeholder(tf.float32, ( 16, roi_size, roi_size, 3), name='inputs_b2h2')
#batch3 hints
inputs_b3h1 = tf.placeholder(tf.float32, ( 16, roi_size, roi_size, 3), name='inputs_b3h1')
inputs_b3h2 = tf.placeholder(tf.float32, ( 16, roi_size, roi_size, 3), name='inputs_b3h2')

inputs_s = tf.placeholder(tf.float32, (None, image_input_size, image_input_size, 3), name='inputs_s')
labels = tf.placeholder(tf.float32,(None,16,2), name='labels')

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
    out_b1h1_l3 =tf.squeeze( tf.cast(tf.sigmoid(out_b1h1_l3), tf.int32))

with tf.variable_scope('conv', reuse=True):
    out_b2h1_l3 = paraNet(inputs_b2h1)
    #flatten and binerize the hashs
    out_b2h1_l3 =tf.squeeze( tf.cast(tf.sigmoid(out_b2h1_l3), tf.int32))

with tf.variable_scope('conv', reuse=True):
    out_b3h1_l3 = paraNet(inputs_b3h1)
    #flatten and binerize the hashs
    out_b3h1_l3 =tf.squeeze( tf.cast(tf.sigmoid(out_b3h1_l3), tf.int32))
    
    #concatenate hint Hash from the 3 batches
    #out_h1_l3 = tf.stack([out_b1h1_l3 , out_b2h1_l3 , out_b3h1_l3])


#network pipeline to create the Second Hint Hash Sets
with tf.variable_scope('conv', reuse=True):
    out_b1h2_l3 = paraNet(inputs_b1h2)
    #flatten and binerize the hashs
    out_b1h2_l3 =tf.squeeze( tf.cast(tf.sigmoid(out_b1h2_l3), tf.int32))

with tf.variable_scope('conv', reuse=True):
    out_b2h2_l3 = paraNet(inputs_b2h2)
    #flatten and binerize the hashs
    out_b2h2_l3 =tf.squeeze( tf.cast(tf.sigmoid(out_b2h2_l3), tf.int32))

with tf.variable_scope('conv', reuse=True):
    out_b3h2_l3 = paraNet(inputs_b3h2)
    #flatten and binerize the hashs
    out_b3h2_l3 =tf.squeeze( tf.cast(tf.sigmoid(out_b3h2_l3), tf.int32))
    #concatenate hint Hash from the 3 batches
    #out_h2_l3 = tf.stack([out_b1h2_l3 , out_b2h2_l3 , out_b3h2_l3])

with tf.variable_scope('conv', reuse=True):
    out_2_l1 = tf.layers.conv2d(inputs_s,  8, [3, 3],strides=(2, 2), padding ='same' ,name='para_conv_1')
    out_2_l1 = tf.nn.relu6(out_2_l1)
    out_2_l2 = tf.layers.conv2d(out_2_l1, 16, [3, 3],strides=(2, 2), padding ='same' ,name='para_conv_2')
    out_2_l2 = tf.nn.relu6(out_2_l2)
    out_2_l3 = tf.layers.conv2d(out_2_l2, 32, [5, 5],strides=(1, 1), padding ='same' ,name='para_conv_3')
    #binerize the value
    out_2_l3 = tf.cast(tf.sigmoid(out_2_l3), tf.int32)
    #iterate through each pixel of the final feature map 
    #turn the value into hashes 
    orig_feature_map_size = tf.shape(out_2_l3)[1]

    loss_batch1, result_batch1 = get_total_loss_and_result(out_2_l3[0] , out_b1h2_l3 , out_b1h1_l3 , orig_feature_map_size , labels[0])
    loss_batch2, result_batch2 = get_total_loss_and_result(out_2_l3[1] , out_b2h2_l3 , out_b2h1_l3 , orig_feature_map_size , labels[1])
    loss_batch3, result_batch3 = get_total_loss_and_result(out_2_l3[2] , out_b3h2_l3 , out_b3h1_l3 , orig_feature_map_size , labels[2])
    loss_of_all_batches = tf.stack([loss_batch1 , loss_batch2 , loss_batch3], axis =0)

#ValueError: No gradients provided for any variable, check your graph for ops that do not support gradients, between variables 
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize( loss_of_all_batches  ,var_list=tf.trainable_variables())


init =  tf.global_variables_initializer()
batchsize = 3

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./variable_graph",graph = sess.graph)
    sess.run(init)
    
    print(tf.trainable_variables())
    #print(hash_set01)
    #print(out_2_l3)

    hintSet01_norm, hintSet02_norm, t_img, t_label_norm = get_train_data(batchsize)
    
    sess.run(train_step , feed_dict={inputs_s:  t_img  , 
                                        inputs_b1h1: hintSet01_norm[0], inputs_b1h2: hintSet02_norm[0], #batch no.1
                                        inputs_b2h1: hintSet01_norm[1], inputs_b2h2: hintSet02_norm[1], #batch no.2
                                        inputs_b3h1: hintSet01_norm[2], inputs_b3h2: hintSet02_norm[2], #batch no.3
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
