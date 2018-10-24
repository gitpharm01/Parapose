import numpy as np
import random
import os
import json

import cv2

def getPaddedROI(img, center_x, center_y, width, height):
    #print(str(int(center_x)) + "," + str(int(center_y)))
    paddingColor = [0,0,0]
    top_left_x = center_x - int(width/2)-1
    #print("top_left_x:")
    #print(top_left_x)
    top_left_y = center_y - int(height/2)-1
    #print("top_left_y:")
    #print(top_left_y)

    bottom_right_x = center_x + int(width/2) 
    bottom_right_y = center_y + int(height/2) 
    #print ("bottom_right_x / y")
    #print(str(bottom_right_x) + " / " + str(bottom_right_y))

    img_height = np.size(img, 0)
    img_width = np.size(img, 1)
    if(top_left_x <0 or top_left_y <0 or bottom_right_x >img_width or bottom_right_y > img_height):
        #border padding needed
        border_left = 0
        border_right = 0
        border_top= 0
        border_bottom= 0

        if(top_left_x < 0):
            width = width + top_left_x
            border_left = -1 * top_left_x
            top_left_x = 0

        if(top_left_y < 0):
            height = height + top_left_y
            border_top = -1 * top_left_y
            top_left_y = 0

        if(bottom_right_x > img_width):
            width = width -(bottom_right_x - img_width)
            border_right = bottom_right_x - img_width
            
        if(bottom_right_y> img_height):
            height = height -(bottom_right_y - img_height)
            border_bottom = bottom_right_y - img_height
        #print(border_left)
        #print(border_right)
        #print(border_top)
        #print(border_bottom)

        img_roi = img[top_left_y : bottom_right_y ,top_left_x : bottom_right_x ]
        #cv2.imshow("originalROI",img_roi)
        img_roi = cv2.copyMakeBorder(img_roi, border_top,border_bottom,border_left, border_right, cv2.BORDER_CONSTANT,value=paddingColor)
    else:
        img_roi = img[top_left_y : bottom_right_y ,top_left_x : bottom_right_x ]
    return img_roi

def training_data_feeder(joint_data_path, train_val_path, imgpath, input_size, hint_roi_size):
    #load trainvalset data,
    train_val = open(train_val_path).readlines()
    train_groups = json.loads(train_val[0].strip())["train_set"]
    #print(train_groups)

    #load one of train set indecies
    index = random.choice(train_groups)
    #print(index)
    #create path object to the image directory( index "0" to dir_name "001")
    dir_name = str(index+1)
    if((index+1) < 100):
        dir_name ="0"+ dir_name 
    if((index+1) < 10):
        dir_name = "0" + dir_name
    #print(dir_name)
    dir_path = imgpath + dir_name + "/"
    #print(dir_path)
    
    
    #ramdomly load three images, get file names
    #from "sample_names" will load first two names as  h_img1  h_iimg2, third name as  t_img
    file_list = []
    for file in os.listdir(dir_path):
        if len(file) > 5:
            file_list.append(file) 
    #print(file_list)
    #print("selected: ")
    sample_name = random.sample(file_list, 3)
    #print(sample_name)
    #load image files
    h_img1 = cv2.imread(dir_path + sample_name[0])
    h_img2 = cv2.imread(dir_path + sample_name[1])
    t_img = cv2.imread(dir_path + sample_name[2])

    #load corresponding joint data as labels
    h_label1 = []
    h_label2 = []
    t_label = []
    
    label_data = open(joint_data_path).readlines()
    for i in range( len(label_data)):
        datum = json.loads(label_data[i].strip())
        if(datum["filename"] == sample_name[0]):
            for joint in datum["joint_pos"]:
                h_label1.append(joint[1])               
            #print(h_label1)            
        elif(datum["filename"] == sample_name[1]):
            for joint in datum["joint_pos"]:
                h_label2.append(joint[1])
        elif(datum["filename"] == sample_name[2]):
            for joint in datum["joint_pos"]:
                t_label.append(joint[1])

    #resize the two images and get resize ratios
    resize_ratioh1 = (input_size / h_img1.shape[1] , input_size / h_img1.shape[0])
    resize_ratioh2 = (input_size / h_img2.shape[1] , input_size / h_img2.shape[0])
    resize_ratiot = (1 / t_img.shape[1] , 1 / t_img.shape[0])
    
    h_img1= cv2.resize(h_img1,(input_size,input_size))
    h_img2= cv2.resize(h_img2,(input_size,input_size))
    t_img = cv2.resize(t_img,(input_size,input_size))
    
    #Convert the joint position according to the resize ratios
    #crop rois from two hint images to get the hintsets
    #img_point = None
    hintSet01 = []
    hintSet02 = []
    
    for joint in h_label1:
        joint[0] = joint[0]*resize_ratioh1[0]
        joint[1] = joint[1]*resize_ratioh1[1]
    for i in range(len(h_label1)):
        tmp = getPaddedROI(h_img1, int(h_label1[i][0]), int(h_label1[i][1]), hint_roi_size, hint_roi_size)
        hintSet01.append(tmp)
        #cv2.imshow("tmp",tmp)
    #cv2.imshow("h_img1",h_img1)
    #for tmp in hintSet01:
    #    cv2.imshow("tmp",tmp)
    #    cv2.waitKey(0)
    for joint in h_label2:
        joint[0] = joint[0]*resize_ratioh2[0]
        joint[1] = joint[1]*resize_ratioh2[1]
    for i in range(len(h_label2)):
        tmp = getPaddedROI(h_img2, int(h_label2[i][0]), int(h_label2[i][1]), hint_roi_size, hint_roi_size)
        hintSet02.append(tmp)
    #Normalize the value by dividing with input_size
    for joint in t_label:
        joint[0] = joint[0]*resize_ratiot[0] 
        joint[1] = joint[1]*resize_ratiot[1] 

    return hintSet01, hintSet02, t_img, t_label    
    
    #cv2.imshow("img_point",img_point)
    #cv2.waitKey(0)
    #cv2.imshow("h_img1",h_img1)
    #cv2.imshow("h_img2",h_img2)
    #cv2.imshow("t_img",t_img)
    #cv2.waitKey(0)

    #define sub function crop roi
    #return roi*16 

#crop rois x 2 times to get 2 hintsets

#return hintset01,hintset02,target image, target label
#joint_data_path = "./custom_data.json"
#train_val_path = "./train_val_indices.json"
#imgpath = "./000/"
#input_size = 400
#hint_roi = 14

#hintSet01,hintSet02,t_img,t_label = training_data_feeder(joint_data_path, train_val_path, imgpath, input_size, hint_roi )
#print(np.shape(hintSet01[0]))
