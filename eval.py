import open3d as o3d
import munkres as mn
import numpy as np
from requests import get
from scipy.spatial import procrustes
#from scipy.linalg import orthogonal_procrustes
import time
from cmath import isnan
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path_gt', default="gt.txt", help='path to groundtruth file')
parser.add_argument('--path_pr', default="pr.txt", help='path to predictions file')
parser.add_argument('--visualization', default=False, type =bool, help='visualize groundtruth and predictions')
parser.add_argument('--visualization_id', default=-1, type=int, help='id of person to visualize')


def zero_out_nans(sk_gt, sk_pr):
    rows, cols = np.shape(sk_gt)

    sk_gt_z = np.zeros([rows, cols])
    sk_pr_z = np.zeros([rows, cols])

    for i in range(0, rows):
        b = False
        for j in range(0, cols):
            b = b or isnan(sk_gt[i][j]) or isnan(sk_pr[i][j])
        if not b:
            sk_gt_z[i] = sk_gt[i]
            sk_pr_z[i] = sk_pr[i]
    
    return sk_gt_z, sk_pr_z


def compute_local_mpjpe(temp_gt, temp_pr):

    pelvis_gt = temp_gt[2]
    pelvis_pr = temp_pr[2]

    if isnan(pelvis_gt[0]) or isnan(pelvis_pr[0]):
        return -1

    rows, cols = np.shape(temp_gt)
    for i in range(0, rows):
        if isnan(temp_gt[i][0]) or isnan(temp_pr[i][0]):
            temp_gt[i] = pelvis_gt
            temp_pr[i] = pelvis_pr

    pelvis_mat_gt = np.repeat(pelvis_gt[:], 17, axis=0).reshape([17, 3], order='F')
    pelvis_mat_pr = np.repeat(pelvis_pr[:], 17, axis=0).reshape([17, 3], order='F')

    temp_gt = temp_gt - pelvis_mat_gt
    temp_pr = temp_pr - pelvis_mat_pr

    return np.sum(np.linalg.norm((temp_gt - temp_pr), axis=1))


def get_pred_id(indices, gt_id):
    for row, col in indices:
        if row == gt_id:
            return col
    return -1

def get_nth_unmatched(indices, n, length):
    count = 0
    mask = np.zeros(length)
    for row, col in indices:
        mask[col]=1

    for i in range(0, length):
        if mask[i] == 0:
            count = count + 1
            
            if count == n:
                return i
    return -1




def visualize(vis, line_set_gt, line_set_pr, skels_gt, skels_pr):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0.5, 0], [0, 1, 0.5], [1, 0, 0.5], [0.5, 1, 0], [0, 0.5, 1], [0.5, 0, 1], 
            [0.33, 0.67, 0.67], [0.67, 0.33, 0.33], [0.33, 0.67, 0.33], [0.33, 0.33, 0.67], [0.67, 0.67, 0.33], [0.33, 0.67, 0.67], [0.67, 0.33, 0.67],
            [0.67, 0.67, 0.67], [0.67, 0, 0], [0, 0.67, 0], [0, 0, 0.67], [0.67, 0.67, 0], [0, 0.67, 0.67], [0.67, 0, 0.67],
            [0.33, 0.33, 0.33], [0.33, 0, 0], [0, 0.33, 0], [0, 0, 0.33], [0.33, 0.33, 0], [0, 0.33, 0.33], [0.33, 0, 0.33]]

    body_lines_pan = [[0, 1], [0, 9], [0, 3], [9, 12], [3, 6], [9, 10], [10, 11], [3, 4], [4, 5], 
                [2, 12], [2, 6], [12, 13], [13, 14], [6, 7], [7, 8], [17, 1], [15, 1], [18, 17], [16, 15]]

    body_lines_skel19 = [[0, 2], [0, 3], [1, 5], [1, 6], [5, 2], [6, 3], [2, 7], [7, 13], [13, 18],
                    [3, 8], [8, 14], [14, 17], [5, 11], [11, 15], [6, 12], [12, 16], [1, 4], [9, 4], [10, 4]]


    num_skels_to_render_gt = len(skels_gt)
    num_skels_to_render_pr = len(skels_pr)
    num_skels_prev_rendered_gt = len(line_set_gt)
    num_skels_prev_rendered_pr = len(line_set_pr)

    for i in range(0, max(num_skels_prev_rendered_pr, num_skels_to_render_pr)):
        if i < min(num_skels_prev_rendered_pr, num_skels_to_render_pr):
            color = [colors[2*i] for j in range(19)]
            line_set_pr[i].points=o3d.utility.Vector3dVector(skels_pr[i])
            line_set_pr[i].lines=o3d.utility.Vector2iVector(body_lines_skel19)
            line_set_pr[i].colors = o3d.utility.Vector3dVector(color)

            vis.update_geometry(line_set_pr[i])
            vis.poll_events()
            vis.update_renderer()
        
        elif i < num_skels_to_render_pr:
            color = [colors[2*i] for j in range(19)]
            line_set_pr.append(o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(skels_pr[i]),
                    lines=o3d.utility.Vector2iVector(body_lines_skel19)
                ))
            line_set_pr[i].colors = o3d.utility.Vector3dVector(color)
            
            vis.add_geometry(line_set_pr[i])
        
        else:
            line_set_pr[i].points=o3d.utility.Vector3dVector([])
            line_set_pr[i].lines=o3d.utility.Vector2iVector([])
            line_set_pr[i].colors = o3d.utility.Vector3dVector([])
            
            vis.update_geometry(line_set_pr[i])
            vis.poll_events()
            vis.update_renderer()

    for i in range(0, max(num_skels_prev_rendered_gt, num_skels_to_render_gt)):
        if i < min(num_skels_prev_rendered_gt, num_skels_to_render_gt):
            color = [colors[2*i+1] for j in range(19)]
            line_set_gt[i].points=o3d.utility.Vector3dVector(skels_gt[i])
            line_set_gt[i].lines=o3d.utility.Vector2iVector(body_lines_pan)
            line_set_gt[i].colors = o3d.utility.Vector3dVector(color)
            vis.update_geometry(line_set_gt[i])
            vis.poll_events()
            vis.update_renderer()

        elif i < num_skels_to_render_gt:
            color = [colors[2*i+1] for j in range(19)]
            line_set_gt.append(o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(skels_gt[i]),
                    lines=o3d.utility.Vector2iVector(body_lines_pan)
                ))
            line_set_gt[i].colors = o3d.utility.Vector3dVector(color)
            vis.add_geometry(line_set_gt[i])

        else:
            line_set_gt[i].points=o3d.utility.Vector3dVector([])
            line_set_gt[i].lines=o3d.utility.Vector2iVector([])
            line_set_gt[i].colors = o3d.utility.Vector3dVector([])
            vis.update_geometry(line_set_gt[i])
            vis.poll_events()
            vis.update_renderer()

    return vis, line_set_gt, line_set_pr



#####################################################
#####################################################
# MAIN FUNCTION
if __name__ == '__main__':
    args = parser.parse_args()

    path_pr = args.path_pr
    path_gt = args.path_gt

    pr = open(path_pr)
    gt = open(path_gt)

    '''pr = open('pr_1.txt')
    gt = open('gt_1.txt')'''

    st_pr = pr.readline()
    sl_pr = st_pr.split()

    st_gt = gt.readline()
    sl_gt = st_gt.split()


    num_kps = min(int(sl_pr[0]), int(sl_gt[0]))
    num_frames = min(int(sl_pr[1]), int(sl_gt[1]))


    mpjpe_glob = 0
    mpjpe_loc = 0
    pa_mpjpe = 0
    num_people_total = 0
    num_people_total_loc = 0


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)

    '''rdr = vis.get_render_option()
    rdr.show_coordinate_frame = True
    vis.poll_events()
    vis.update_renderer()'''

    line_set_pr = []
    line_set_gt = []


    for i in range(0, num_frames):

        num_people_pr = int(pr.readline())
        num_people_gt = int(gt.readline())

        num_people_total = num_people_total + min(num_people_gt, num_people_pr)
        num_people_total_loc = num_people_total_loc + min(num_people_gt, num_people_pr)

        #these skel coordinates are used for computing the mpjpe errors
        #they consist only of those joints that are shared between the joint formats of gt and predictions
        skels_pr = np.zeros([num_people_pr, 17, 3])
        skels_gt = np.zeros([num_people_gt, 17, 3])

        #those skel coordinates are used for visualizations
        skels_pr_original_size = np.zeros([num_people_pr, 19, 3])
        skels_gt_original_size = np.zeros([num_people_gt, 19, 3])


        ###################################################################
        #read in joint coordinates of people from the predictions .txt file
        for j in range(0, num_people_pr):
            id = int(pr.readline())

            xs_str = pr.readline()
            ys_str = pr.readline()
            zs_str = pr.readline()
            probs_str = pr.readline()

            xs = xs_str.split()
            ys = ys_str.split()
            zs = zs_str.split()
            probs = probs_str.split()

            skels_pr_temp = np.zeros([num_kps, 3])

            for k in range(0, num_kps):
                
                if float(probs[k]) != -1:
                    skels_pr_temp[k][0] = float(xs[k]) *1000
                    skels_pr_temp[k][1] = float(ys[k]) *1000
                    skels_pr_temp[k][2] = float(zs[k]) *1000
                    skels_pr_original_size[j][k][0] = float(xs[k]) *1000
                    skels_pr_original_size[j][k][1] = float(ys[k]) *1000
                    skels_pr_original_size[j][k][2] = float(zs[k]) *1000
                else:
                    skels_pr_temp[k][0] = np.NaN
                    skels_pr_temp[k][1] = np.NaN
                    skels_pr_temp[k][2] = np.NaN
                    skels_pr_original_size[j][k][0] = np.NaN
                    skels_pr_original_size[j][k][1] = np.NaN
                    skels_pr_original_size[j][k][2] = np.NaN

            #shuffle joint coordinates so that the joints in skels_pr are in the same order as in the groundtruth skeletons skels_gt
            skels_pr[j][0] = skels_pr_temp[1]
            skels_pr[j][1] = skels_pr_temp[4]
            skels_pr[j][2] = skels_pr_temp[0]
            skels_pr[j][3] = skels_pr_temp[6]
            skels_pr[j][4] = skels_pr_temp[12]
            skels_pr[j][5] = skels_pr_temp[16]
            skels_pr[j][6] = skels_pr_temp[3]
            skels_pr[j][7] = skels_pr_temp[8]
            skels_pr[j][8] = skels_pr_temp[14]
            skels_pr[j][9] = skels_pr_temp[5]
            skels_pr[j][10] = skels_pr_temp[11]
            skels_pr[j][11] = skels_pr_temp[15]
            skels_pr[j][12] = skels_pr_temp[2]
            skels_pr[j][13] = skels_pr_temp[7]
            skels_pr[j][14] = skels_pr_temp[13]
            skels_pr[j][15] = skels_pr_temp[10]
            skels_pr[j][16] = skels_pr_temp[9]


        ###################################################################
        #read in joint coordinates of people from the predictions .txt file
        for j in range(0, num_people_gt):
            id = int(gt.readline())

            xs_str = gt.readline()
            ys_str = gt.readline()
            zs_str = gt.readline()
            probs_str = gt.readline()

            xs = xs_str.split()
            ys = ys_str.split()
            zs = zs_str.split()
            probs = probs_str.split()

            for k in range(0, 15):

                if float(probs[k]) != -1:
                    skels_gt[j][k][0] = float(xs[k]) *1000
                    skels_gt[j][k][1] = float(ys[k]) *1000
                    skels_gt[j][k][2] = float(zs[k]) *1000
                else:
                    skels_gt[j][k][0] = np.NaN
                    skels_gt[j][k][1] = np.NaN
                    skels_gt[j][k][2] = np.NaN

            if float(probs[15]) != -1:
                skels_gt[j][15][0] = float(xs[16]) *1000
                skels_gt[j][15][1] = float(ys[16]) *1000
                skels_gt[j][15][2] = float(zs[16]) *1000
            else:
                skels_gt[j][15][0] = np.NaN
                skels_gt[j][15][1] = np.NaN
                skels_gt[j][15][2] = np.NaN

            if float(probs[16]) != -1:
                skels_gt[j][16][0] = float(xs[18]) *1000
                skels_gt[j][16][1] = float(ys[18]) *1000
                skels_gt[j][16][2] = float(zs[18]) *1000
            else:
                skels_gt[j][16][0] = np.NaN
                skels_gt[j][16][1] = np.NaN
                skels_gt[j][16][2] = np.NaN

            for k in range(0, 19):
                if float(probs[k]) != -1:
                    skels_gt_original_size[j][k][0] = float(xs[k]) *1000
                    skels_gt_original_size[j][k][1] = float(ys[k]) *1000
                    skels_gt_original_size[j][k][2] = float(zs[k]) *1000
                    if float(xs[k])== 0 and float(ys[k])==0 and float(zs[k])==0:
                        print("here")
                else:
                    skels_gt_original_size[j][k][0] = np.NaN
                    skels_gt_original_size[j][k][1] = np.NaN
                    skels_gt_original_size[j][k][2] = np.NaN

        ######################################################################################################
        #EVALUATION
        if ((num_people_gt!=0) and (num_people_pr!=0)):
        
            ######################################################################################################
            #match the skeletons contained in the prediction and the groundtruth files via the Hungarian Algorithm
            mat = np.zeros([num_people_gt, num_people_pr])

            for j in range(0, num_people_gt):
                for k in range(0, num_people_pr):
                    skels_gt_z, skels_pr_z = zero_out_nans(skels_gt[j], skels_pr[k])
                    mat[j][k] = np.sum(np.linalg.norm((skels_gt_z - skels_pr_z), axis=1))
            if i == 2400:
                print(mat)
                print()
            mat_ = np.copy(mat)

            if j > k:
                mat_ = np.transpose(mat_)

            m = mn.Munkres()
            indexes = m.compute(mat_)
            
            if j > k:
                l = []
                for row, column in indexes:
                    l.append((column, row))
                indexes = l
            

            ################################################
            #compute global mpjpe value of the current frame
            err = 0

            for row, column in indexes:
                value = mat[row][column] / 17
                err += value
            
            if i == 2400:
                print(mat)

            mpjpe_glob += err

            
            ###############################################
            #compute local mpjpe value of the current frame
            err1 = 0

            for row, column in indexes:
                temp_gt = skels_gt[row].copy()
                temp_pr = skels_pr[column].copy()

                error_norm = compute_local_mpjpe(temp_gt, temp_pr) / 17
                
                if error_norm != -1:
                    err1 += error_norm
                    #print(str(error_norm))
                else:
                    num_people_total_loc -= 1

            mpjpe_loc += err1

            
            ############################################################
            #compute procrustes aligned mpjpe value of the current frame
            err2 = 0

            for row, column in indexes:
                temp_gt = skels_gt[row].copy()
                temp_pr = skels_pr[column].copy()

                rows, cols = np.shape(temp_gt)
                r, c = np.shape(temp_pr)

                cnt = 0
                for i in range(0, rows):
                    if isnan(skels_gt[row][i][0]) or isnan(skels_pr[column][i][0]):
                        temp_gt = np.delete(temp_gt, cnt, 0)
                        temp_pr = np.delete(temp_pr, cnt, 0)
                    else:
                        cnt+=1

                rows, cols = np.shape(temp_gt)
                r, c = np.shape(temp_pr)

                if (((rows>0) and (cols>0)) and ((r>0) and (c>0))):

                    mtx1, mtx2, disparity = procrustes(temp_gt, temp_pr)
                    #mtx1, vals = orthogonal_procrustes(temp_gt, temp_pr)
                    #disparity = compute_local_mpjpe(np.matmul(temp_gt, mtx1), temp_pr)
                    err2 += (disparity / 17)
            
            pa_mpjpe += err2
            

        
        ######################
        #create visualizations
        if args.visualization:
            print(str(args.visualization)) 
            #vis_id = -1
            vis_id = args.visualization_id

            if vis_id== -1:
                vis, line_set_gt, line_set_pr = visualize(vis, line_set_gt, line_set_pr, skels_gt_original_size, skels_pr_original_size)
                #time.sleep(0.0001)
            
            elif vis_id < len(skels_gt):
                pred_id = get_pred_id(indexes, vis_id)

                if pred_id != -1:
                    vis, line_set_gt, line_set_pr = visualize(vis, line_set_gt, line_set_pr, skels_gt_original_size[vis_id:(vis_id+1), :, :], skels_pr_original_size[pred_id:(pred_id+1), :, :])
                    #time.sleep(0.1)
                
                else:
                    vis, line_set_gt, line_set_pr = visualize(vis, line_set_gt, line_set_pr, skels_gt_original_size[vis_id:(vis_id+1), :, :], [])
                    #time.sleep(0.1)

            else:
                pred_id = get_nth_unmatched(indexes, (vis_id-len(skels_gt)+1), len(skels_pr))
                
                if pred_id != -1:
                    vis, line_set_gt, line_set_pr = visualize(vis, line_set_gt, line_set_pr, [], skels_pr_original_size[pred_id:(pred_id+1), :, :])
                    #time.sleep(0.1)
                else:
                    print("cannot visualize, too few people detected")
        


    ##################################################
    #compute avarage global and local mpjpe per frames
    if num_people_total !=0:
        mpjpe_glob = mpjpe_glob / num_people_total
        pa_mpjpe = pa_mpjpe / num_people_total
        print("global mpjpe result: "+str(mpjpe_glob))
        #print("procrustes mpjpe result: "+str(pa_mpjpe))
    if num_people_total_loc !=0:  
        mpjpe_loc = mpjpe_loc / num_people_total_loc
        print("local mpjpe result: "+str(mpjpe_loc))

    vis.destroy_window()
