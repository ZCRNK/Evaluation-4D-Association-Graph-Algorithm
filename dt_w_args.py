import array
import json
import numpy as np
import math
import os, os.path
import argparse

from matplotlib import image
from matplotlib import pyplot as plt
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--path_joints', default="/media/zreinke/My Passport/joints", help='path to joint keypoint.json file')
parser.add_argument('--path_pafs', default="/media/zreinke/My Passport/pafs", help='path to paf.float file')
parser.add_argument('--path_t', default="../170407_haggling_a1/0.txt", help='path to the target file')
parser.add_argument('--name_beginning', default="0_", help='first number of each name of the joint keypoint files')


#from https://github.com/zhangyux15/4d_association/issues/17
#adapted according to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/net/bodyPartConnectorBase.cpp
def get_score(x1, y1, x2, y2, pafMatX, pafMatY, cnt):

    dx, dy = x2 - x1, y2 - y1
    normVec = math.sqrt(dx ** 2 + dy ** 2)

    max_d = max(abs(dx), abs(dy))
    num_inter = max(5, min(25, round(math.sqrt(5*max_d))))

    if normVec < 1e-5:
        return 0.0, 0

    vx, vy = dx / normVec, dy / normVec

    xs = np.arange(x1, x2, dx / num_inter) if x1 != x2 else np.full((num_inter, ), x1)
    ys = np.arange(y1, y2, dy / num_inter) if y1 != y2 else np.full((num_inter, ), y1)

    xs = np.append(xs, [x2])
    ys = np.append(ys, [y2])

    xs = (xs + 0.5).astype(np.int_)
    ys = (ys + 0.5).astype(np.int_)

    pafXs = np.zeros(num_inter+2)
    pafYs = np.zeros(num_inter+2)

    rows, cols = pafMatX.shape

    for idx, (mx, my) in enumerate(zip(xs, ys)):
        if my < rows and mx < cols and idx < len(pafXs):
            pafXs[idx] = pafMatX[my][mx]
            pafYs[idx] = pafMatY[my][mx]
        if my < 0 or mx < 0:
            print(str(x1)+"    "+str(y1)+"    "+str(x2)+"    "+str(y2))
            print(str(mx)+"    "+str(my))

    local_scores = pafXs * vx + pafYs * vy

    thidxs = local_scores > 0.05

    if ((sum(thidxs))/num_inter) > 0.85:
        return round((sum(np.multiply(local_scores, thidxs)))/ (sum(thidxs)), 6)
    
    else:
        return 0


#function for computing keypoints and association scores and writing them to the output file
def compute_keypoints_and_association_scores(data, x, t):
    
    ##READ IN KEYPOINTS DATA
    #initialize array of size 25 X 3 X 1 (#body_parts X #dims X #detected_persons)
    arr= [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], 
            [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],
            [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],
            [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]],
            [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]]


    #fill array

    if data.get("part_candidates") is not None:
        for i in range(0, 25):
            kps = data.get("part_candidates")[0].get(str(i))

            l2 = int(len(kps)/3)

            for j in range (0, l2):
                if kps[3*j+2] > 0.17:
                    arr[i][0].append(kps[3*j])
                    arr[i][1].append(kps[3*j+1])
                    arr[i][2].append(kps[3*j+2])
   

    ##WRITE KEYPOINTS DATA

    for i in range (0, 25):
        l2 = len(arr[i][0])
        t.write(str(l2)+"\n")
        
        for j in range(0, l2):
            t.write(str(arr[i][0][j])+"\t")
        t.write("\n")

        for j in range(0, l2):
            t.write(str(arr[i][1][j])+"\t")
        t.write("\n")
        
        for j in range(0, l2):
            t.write(str(arr[i][2][j])+"\t")
        t.write("\n")



    ##COMPUTE AND WRITE ASSOCIATION SCORES
    assert x[0] == 3 # First parameter saves the number of dimensions (18x300x500 = 3 dimensions)

    shape_x = x[1:1+int(x[0])]
    d1 = int(shape_x[0]) # Size of the first dimension = number PAFS
    d2 = int(shape_x[1]) # Size of the second dimension = y-direction
    d3 = int(shape_x[2]) # Size of the third dimension = x-direction
    pafs = x[1+int(round(x[0])):]


    #pafs = pafs.reshape(d2, d1*d3, order='F')
    pafs = pafs.reshape(d1, d2, d3, order='C')

    #define the body pairs according to skeleton BODY_25
    #_skelDefs[BODY25].pafDict <<1, 9,  10, 8, 8,  12, 13, 1, 2, 3, 2,  1, 5, 6, 5,  1, 0,   0, 15, 16, 14, 19, 14, 11, 22, 11,
	#    	                     8, 10, 11, 9, 12, 13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24

    pairs = [[1,8, 0], [9,10, 1], [10,11, 2], [8,9, 3], [8,12, 4], [12,13, 5], [13,14, 6], [1,2, 7], [2,3, 8], [3,4, 9], [2,17, 10], [1,5, 11], [5,6, 12], [6,7, 13], [5,18, 14], [1,0, 15], 
            [0,15, 16], [0,16, 17], [15,17, 18], [16,18, 19], [14,19, 20], [19,20, 21], [14,21, 22], [11,22, 23], [22,23, 24], [11,24, 25]]

    l_pairs = len(pairs)

    counter=0


    for i in range(0, l_pairs):
        
        p1 = pairs[i][0]
        p2 = pairs[i][1]
        p3 = pairs[i][2]

        lp1= len(arr[p1][0])
        lp2= len(arr[p2][0])
        
        #iterate through p1
        for j in range(0, lp1):
            #iterate through p2
            for k in range (0, lp2):
                #paf_ind = i
                paf_ind = p3

                #get x and y values and scale them (original x and y values are between 0 and 1)
                x1= arr[p1][0][j] * d3
                x2= arr[p2][0][k] * d3
                y1= arr[p1][1][j] * d2
                y2= arr[p2][1][k] * d2

                paf_x = pafs[2*paf_ind]
                paf_y = pafs[2*paf_ind+1]

                score = get_score(x1, y1, x2, y2, paf_x, paf_y, counter)
                counter =counter +1

                t.write(str(score)+"\t")
            
            t.write("\n")
            



## main function
if __name__ == '__main__':
    args = parser.parse_args()

    path_joints = args.path_joints
    path_pafs = args.path_pafs
    path_t = args.path_t
    name_beginning = args.name_beginning


    num_files_k = len([name for name in os.listdir(path_joints)])
    num_files_h = len([name for name in os.listdir(path_pafs)])
    #assert num_files_h == num_files_k
    print(str(num_files_h))

    t = open(path_t, 'w')

    t.write(str(4)+ "\t")
    t.write(str(num_files_h)+ "\n")

    for i in range(0, num_files_k):

        #construct file name
        le = len(str(i))
        nd = 12 - le

        fn = name_beginning
        for j in range(0, nd):
            fn = fn + "0"

        fn_k = path_joints + "/" + fn + str(i) + "_keypoints.json"
        fn_h = path_pafs + "/" + fn + str(i) + "_pose_heatmaps.float"

        #open JSON file
        f = open(fn_k)
        #return JSON object as a dictionary
        data = json.load(f)

        #open heatmap file
        x = np.fromfile(fn_h, dtype=np.float32)

        #compute and write keypoints and association scores
        compute_keypoints_and_association_scores(data, x, t)
