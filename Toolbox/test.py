#***************************************************************************
#* 
#* Description: label propagation
#* Author: Zou Xiaoyi (zouxy09@qq.com)
#* Date:   2015-10-15
#* HomePage: http://blog.csdn.net/zouxy09
#* 
#**************************************************************************
 
import time
import math
import numpy as np
from label_propagation import labelPropagation
#from torch.nn import functional as F
import os
# show
def show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels): 
    import matplotlib.pyplot as plt 
    
    for i in range(Mat_Label.shape[0]):
        if int(labels[i]) == 0:  
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dr')  
        elif int(labels[i]) == 1:  
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Db')
        else:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dy')
    
    for i in range(Mat_Unlabel.shape[0]):
        if int(unlabel_data_labels[i]) == 0:  
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'or')  
        elif int(unlabel_data_labels[i]) == 1:  
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'ob')
        else:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'oy')
    
    plt.xlabel('X1'); plt.ylabel('X2') 
    plt.xlim(0.0, 1.)
    plt.ylim(0.0, 1.)
    plt.show()  
 
def showNus(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels):
    import matplotlib.pyplot as plt 
    Mat = np.concatenate((Mat_Label,Mat_Unlabel), axis=0)
    label = np.concatenate((labels, unlabel_data_labels), axis=0)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(Mat_Unlabel[:,1],Mat_Unlabel[:,2],Mat_Unlabel[:,0],c=unlabel_data_labels, s=5)

    ax.scatter3D(Mat_Label[:20,1], Mat_Label[:20,2], Mat_Label[:20, 0], c=labels, s=50)
    
    #设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Logit')
    plt.xlim(0.0, 1.)
    plt.ylim(1.0, 0.0)
    plt.show()   

def loadCircleData(num_data):
    center = np.array([5.0, 5.0])
    radiu_inner = 2
    radiu_outer = 4
    num_inner = num_data / 3
    num_outer = num_data - num_inner
    
    data = []
    theta = 0.0
    for i in range(int(num_inner)):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_inner * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_inner * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 2
    
    theta = 0.0
    for i in range(int(num_outer)):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_outer * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_outer * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 1
    
    Mat_Label = np.zeros((2, 2), np.float32)
    Mat_Label[0] = center + np.array([-radiu_inner + 0.5, 0])
    Mat_Label[1] = center + np.array([-radiu_outer + 0.5, 0])
    labels = [0, 1]
    Mat_Unlabel = np.vstack(data)
    return Mat_Label, labels, Mat_Unlabel
 
 
def loadBandData(num_unlabel_samples):
    #Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])
    #labels = [0, 1]
    #Mat_Unlabel = np.array([[5.1, 2.], [5.0, 8.1]])
    
    Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])
    labels = [0, 1]
    num_dim = Mat_Label.shape[1]
    Mat_Unlabel = np.zeros((num_unlabel_samples, num_dim), np.float32)
    Mat_Unlabel[:num_unlabel_samples/2, :] = (np.random.rand(int(num_unlabel_samples/2), num_dim) - 0.5) * np.array([3, 1]) + Mat_Label[0]
    Mat_Unlabel[num_unlabel_samples/2 : num_unlabel_samples, :] = (np.random.rand(int(num_unlabel_samples/2), num_dim) - 0.5) * np.array([3, 1]) + Mat_Label[1]
    return Mat_Label, labels, Mat_Unlabel
 

def loadNusData():
    root_path = os.path.join(os.environ['HOME'], 'Work_dir/weaklysup/tutorial/Toolbox')
    pred_mask = np.load(os.path.join(root_path, 'pred_mask.npy'))
    feat = np.load(os.path.join(root_path,'feat.npy'))
    lpoint_coord = np.load(os.path.join(root_path,'point_coord.npy'))
    lpoint_label = np.load(os.path.join(root_path,'point_label.npy'))
    #(28, 28) (256, 14, 14) (20, 2) (20,)

    feat_dim, feat_size, logit_size = feat.shape[0], feat.shape[1], pred_mask.shape[0]
    num_lable_samples = lpoint_coord.shape[0]
    num_unlabel_samples = logit_size * logit_size
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    Mat_label = []
    for i in range(num_lable_samples):
        # use feat to cal Euclidistance
        # point_feat = feat[:, int(lpoint_coord[i, 0]*feat_size), int(lpoint_coord[i, 1]*feat_size)]
        point_logit = pred_mask[int(lpoint_coord[i, 0]*logit_size), int(lpoint_coord[i, 1]*logit_size)]
        Mat_label.append([sigmoid(point_logit), lpoint_coord[i, 0], lpoint_coord[i, 1]])
    Mat_label = np.array(Mat_label)

    label = lpoint_label
    Mat_unlabel = []
    for i in range(num_unlabel_samples):
        x, y = i//logit_size, i%logit_size
        unlabel_feat = [sigmoid(pred_mask[int(x), int(y)]), x/logit_size, y/logit_size]
        Mat_unlabel.append(unlabel_feat)
    #Mat_unlabel = feat.reshape(num_unlabel_samples, -1)
    Mat_unlabel = np.array(Mat_unlabel)
    return Mat_label, label, Mat_unlabel



# main function
if __name__ == "__main__":
    num_unlabel_samples = 800
    #Mat_Label, labels, Mat_Unlabel = loadBandData(num_unlabel_samples)
    #Mat_Label, labels, Mat_Unlabel = loadCircleData(num_unlabel_samples)
    Mat_Label, labels, Mat_Unlabel = loadNusData()
    
    ## Notice: when use 'rbf' as our kernel, the choice of hyper parameter 'sigma' is very import! It should be
    ## chose according to your dataset, specific the distance of two data points. I think it should ensure that
    ## each point has about 10 knn or w_i,j is large enough. It also influence the speed of converge. So, may be
    ## 'knn' kernel is better!
    #unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 0.2)
    unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'knn', knn_num_neighbors = 20, max_iter = 1000)
    #unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 0.025)

    #show(Mat_Label[:,:2], labels, Mat_Unlabel[:,:2], unlabel_data_labels)

    showNus(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels)
