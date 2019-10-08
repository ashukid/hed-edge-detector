import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--prototxt', help='Path to deploy.prototxt', required=True)
parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel', required=True)
parser.add_argument('--width', help='Resize input image to a specific width', default=256, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=256, type=int)
parser.add_argument('--savefile', help='Specifies the output video path', default='output.mp4', type=str)
args = parser.parse_args()

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

# Load the model.
net = cv.dnn.readNetFromCaffe(args.prototxt, args.caffemodel)
cv.dnn_registerLayer('Crop', CropLayer)

kWinName = 'Holistically-Nested_Edge_Detection'
cv.namedWindow('Input', cv.WINDOW_AUTOSIZE)
cv.namedWindow(kWinName, cv.WINDOW_AUTOSIZE)
cv.namedWindow("Canny", cv.WINDOW_AUTOSIZE)

cap = cv.VideoCapture(args.input if args.input else 0)
WRITE_VIDEO_FLAG=True
if WRITE_VIDEO_FLAG:
    # Define the codec and create VideoWriter object
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(w,h)
    # w, h = args.width,args.height
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    writer = cv.VideoWriter(args.savefile, fourcc, 25, (w, h))
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    #cv.imshow('Input', frame)
    # width,height = frame
    inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(args.width, args.height),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    # edges = cv.Canny(frame,args.width,args.height)
    edges = cv.Canny(frame,frame.shape[1],frame.shape[0])
    out = net.forward()
    # print(out.shape)
    # print(frame[0][0][0])
    # print(out)
    out = out[0, 0]
    out = cv.resize(out, (frame.shape[1], frame.shape[0]))
    # print(frame.shape[1], frame.shape[0])
    # cv.imwrite("ouuut.jpg",out)
    # f=cv.cvtColor(f,cv.COLOR_BGR2GRAY)
    print(out.shape)
    out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
    # blur = cv.GaussianBlur(out,(5,5),0)
    # ret,out = cv.threshold(out,0.5 ,255,cv.THRESH_BINARY)
    # frame = cv.medianBlur(frame,5)
    # cv.imwrite("ouuut.jpg",frame)
    # ret,frame=cv.threshold(frame,127,255,cv.THRESH_BINARY)
    # frame = cv.adaptiveThreshold(frame,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
    # cv.imwrite("ouuut.jpg",frame)
    # print(out[0][0][0])
    #out = (out < 100) * out
    #np.clip(out, 0, 1, out=out)
    #out=out *255
    #out[out >= 1000] = 255
    #info = np.finfo(out.dtype)
    #print(info.max)
    #print(out)
    #out = out.astype(np.float64) / info.max 
    out = 255 * out
    # print(out)
    out = out.astype(np.uint8)
    # out = cv.fromArray(out)
    print(type(out))
    print(np.max(out))
    print(np.min(out))
    print(out.shape)
    print(frame.shape)
    # frame = frame.astype(np.uint8)
    # out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
    # print(out.shape)
    #concated=np.hstack((out,f))
    #con=np.concatenate([edges,out],axis=1)
    con=np.concatenate((frame,out),axis=1)
    ##cv.imshow("Canny", con)
    cv.imshow(kWinName, out)
    if args.input:
        writer.write(np.uint8(con))
    #else:
    #    cv.imwrite("out.mp4",out)
    #gt = cv.imread("gt-"+args.input)
    # print(frame.shape)
#     cv.imshow('Input',frame)
    #cv.imshow('Human Annotated',gt)
