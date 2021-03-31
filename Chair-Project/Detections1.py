# My understanding is that I need to use face detection only for Human ROI, for
# this particular approach. Also, I need to play around with the code. See for
# Myself. I must plot all Chair and Human ROIs too, in each approach.
import tensorflow as tf
import cv2 as cv
import sys
import os

image = ''
video = ''

def IOU(c,p):
    px = (p[2] - p[0])/2 + p[0]
    py = (p[3] - p[1])/2 + p[1]
    lx,ux = c[0] + (c[2] - c[0])/3,c[0] + 2*(c[2]-c[0])/3
    ly,uy = max(0,c[1] - (c[3] - c[1])/2),c[1] + (c[3]-c[1])/3
    if px < ux and px > lx:
        if py < uy and py > ly:
            return 1
    return 0

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
    while cv.waitKey(1) < 0: # use ord('q') stuff here. Take idea from Rahat Repo.
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
        chair_list = []
        ctr = 0
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classID = int(out[3][0][i])
            bbox = [float (v) for v in out[2][0][i]]
            score = float(out[1][0][i])
            if classID == 1 or classID == 62: # Confirm classids here from other program.
                if score > 0.3:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    if classID == 62:
                        #regions[app[0]][app[1]].append([x,y,right,bottom])
                        chair_list.append([x,y,right,bottom])
                        ctr += 1
                    if classID == 1:
                        for item in chair_list:#for item in regions[app[0]][app[1]]: # for item in chair_list:
                            r = IOU(item,[x,y,right,bottom])
                            if r == 1:
                                positives.append(item)
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
