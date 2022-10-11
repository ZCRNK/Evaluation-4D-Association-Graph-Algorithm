import os, os.path
import cv2
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seq_name', default="170407_haggling_a1", type=str, help='name of the current video sequence')
parser.add_argument('--fr_num', default=10, type=int, help="total frame number of the output video")
parser.add_argument('--fr_offset', default=10, type=int, help="frame number at which the cut video should start")
parser.add_argument('--ds_rate', default=10, type=int, help="rate by which the video will get downsampled")

if __name__ == '__main__':
    args = parser.parse_args()

    seq_name = args.seq_name
    ds_rate = args.ds_rate
    fr_num = args.fr_num
    fr_offset = args.fr_offset

    #seq_name = "170407_haggling_a1"

    count = 0
    frame_min = 0
    frame_max = 0
    path_gt = "../panoptic-toolbox/"+ seq_name + "/hdPose3d_stage1_coco19/*.json"

    path_imgs = "../panoptic-toolbox/"+ str(seq_name) + "/hdImgs/00_00"
    num_imgs = len([name for name in os.listdir(path_imgs)])


    filelist = glob.glob(os.path.join('', path_gt))
    print(str(len(filelist)))
    for fl in sorted(filelist, key=lambda s: s.lower()):
        num = int(fl.split("_", 5)[5].split(".", 1)[0])

        if count == 0:
            frame_min = num
        
        if (((num % ds_rate != 0) or (num >= num_imgs)) or ((num >= (fr_num + fr_offset)) or (num < fr_offset))):

            #path0 = "../panoptic-toolbox/"+ seq_name + "/hdPose3d_stage1_coco19/" + fl + ".json"
            os.remove(fl)
        
        count += 1
        frame_max = num



    for i in range(0, num_imgs):

        #construct file name
        le = len(str(i))
        nd = 8 - le

        fn0 = "00_00_"
        fn1 = "00_01_"
        fn2 = "00_02_"
        fn3 = "00_03_"
        fn4 = "00_04_"

        for j in range(0, nd):
            fn0 = fn0 + "0"
            fn1 = fn1 + "0"
            fn2 = fn2 + "0"
            fn3 = fn3 + "0"
            fn4 = fn4 + "0"
        
        if (((i % ds_rate != 0) or ((i < frame_min) or (i > frame_max)))) or ((i >= (fr_num + fr_offset)) or (i < fr_offset)):
            '''if i < frame_min:
                print("cond1 "+str(i))
            
            if i > frame_max:
                print("cond2 "+str(i))'''

            path0 = "../panoptic-toolbox/"+ seq_name + "/hdImgs/00_00/" + fn0 + str(i) + ".jpg"
            path1 = "../panoptic-toolbox/"+ seq_name + "/hdImgs/00_01/" + fn1 + str(i) + ".jpg"
            path2 = "../panoptic-toolbox/"+ seq_name + "/hdImgs/00_02/" + fn2 + str(i) + ".jpg"
            path3 = "../panoptic-toolbox/"+ seq_name + "/hdImgs/00_03/" + fn3 + str(i) + ".jpg"
            path4 = "../panoptic-toolbox/"+ seq_name + "/hdImgs/00_04/" + fn4 + str(i) + ".jpg"

            if os.path.exists(path0):
                os.remove(path0)
            if os.path.exists(path1):
                os.remove(path1)
            if os.path.exists(path2):
                os.remove(path2)
            if os.path.exists(path3):
                os.remove(path3)
            if os.path.exists(path4):
                os.remove(path4)

    

    # https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
    for i in range (0, 5):
        image_folder = "../panoptic-toolbox/"+ seq_name + "/hdImgs/00_0" + str(i)
        video_name = "../panoptic-toolbox/"+ seq_name + "/hdVideos/" + str(i) + ".avi"

        images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")], key=lambda s: s.lower())

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        fr = int(30/ ds_rate)

        video = cv2.VideoWriter(video_name, 0, fr, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

