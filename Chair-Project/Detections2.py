import tensorflow as tf
import cv2 as cv
#import argparse
import sys
import os
#import numpy as np

'''parser = argparse.ArgumentParser(description='Use this script to run Mask-RCNN object detection and segmentation')
parser.add_argument('--image', help='Path to image file')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()'''
image = ''
video = ''
def IOU(a,b):
    if len(a) == 0 or len(b) == 0:
        return 0
    xA = max(a[0],b[0])
    yA = max(a[1],b[1])
    xB = min(a[2],b[2])
    yB = min(a[3],b[3])
    i = max(0,xB-xA+1)*max(0,yB-yA +1)
    A = (a[2] - a[0] + 1)*(a[3] - a[1] + 1)
    B = (b[2] - b[0] + 1)*(b[3] - b[1] + 1)
    iou = i/float(A + B - i)
    return iou
    '''
    x2,y2,right2,bottom2 = a # check if individual assignments need to be done
    x1,y1,right1,bottom1 = b
    if x2-x1 > 0:
        if x2 - right1 < 0:
            l = right1 - x2
        else:
            l = 0
    elif right2-x1 > 0:
        l = right2 - x1
    else:
        l = 0
    if y2-y1 > 0:
        if y2 - bottom1 < 0:
            b = bottom1 - y2
        else:
            b = 0
    elif bottom2-y1 > 0:
        b = bottom2 - y1
    else:
        b = 0
    i = l*b
    u = (right2-x2)*(bottom2 - y2) + (right1 - x1)*(bottom1 - y1) - i
    return i/u
    '''   
def assign(x,y,rows,cols):
    cx,cy = cols/3,rows/3
    j,i = int(x//cx),int(y//cy)
    return (i,j)
winName = 'Chairs'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
with tf.gfile.FastGFile(r"frozen_inference_graph.pb","rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    if len(image) != 0:#args.image:
        inputfile = image#args.image
        if not os.path.isfile(image):#args.image):
            print("Input image file ", image, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(image)#cv.imread(inputfile)
    elif len(video) != 0:#args.video:
        inputfile = video #args.video
        if not os.path.isfile(video):#args.video):
            print("Input Video file", video, "does not exist")
            sys.exit(1)
        cap = cv.VideoCapture(video)#args.video)
    else:
        cap = cv.VideoCapture(0)
        inputfile = "camvid.avi"
    inp = list(inputfile.split("\\"))[-1]
    output_file = "Out\\" + inp
    if len(image) == 0:#(not args.image):
        vid_writer = cv.VideoWriter(output_file, cv.VideoWriter_fourcc('M','J','P','G'), 28, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
    
    
    # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", output_file)
            cv.waitKey(3000)
            break
        img = frame
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
        positives = []
        people = []
        #chair_list = []
        regions = [[[] for j in range(3)] for i in range(3)]
        ctr = 0
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classID = int(out[3][0][i])
            bbox = [float (v) for v in out[2][0][i]]
            score = float(out[1][0][i])
            if classID == 1 or classID == 62: 
                if score > 0.3:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    w,h = right - x,bottom - y
                    #app = assign(x + w/2,y+h/2,h,w)
                    app  = assign(x + w/2,y+h/2,rows,cols) # 300, 300 ?
                    if classID == 62:
                        regions[app[0]][app[1]].append([x,y,right,bottom])
                        #chair_list.append([x,y,right,bottom])
                        ctr += 1
                    if classID == 1:
                        people.append([[x,y,right,bottom],[app[0],app[1]]])
        for person in people:
           cv.rectangle(img, (int(person[0][0]),int(person[0][1])),(int(person[0][2]),int(person[0][3])),(255,125,51),thickness = 2)
           label = "person"
           cv.putText(img,label,(int(person[0][0]),int(person[0][1])),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
           for chair in regions[person[1][0]][person[1][1]]:
               r = IOU(chair,person[0])
               if r >= 0.5:
                   positives.append(chair)
        for i in range(len(regions)):
            for j in range(len(regions[0])):
                for chair in regions[i][j]:
                    if len(chair) != 0:
                        cv.rectangle(img,(int(chair[0]), int(chair[1])), (int(chair[2]), int(chair[3])), (125, 51, 255), thickness=2)
                        label = "chair"
                        cv.putText(img, label, (int(chair[0]), int(chair[1])),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        for item in positives:
            cv.rectangle(img, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (125, 255, 51), thickness=2)
            label = "Occupied Chair"
            cv.putText(img, label, (int(item[0]), int(item[1])),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        label = "Total chairs: {0}, vacant chairs: {1}".format(ctr,ctr - len(positives))
        cv.putText(img,label,(15,15),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255))
        #cv.imwrite(output_file,img.astype(np.uint8)) # Will this work even for a video ? Check out opencv video editing, and make necessary modifications.
        cv.imshow(winName,img)
        #cv.waitKey(5000)
cv.destroyAllWindows()
cap.release
