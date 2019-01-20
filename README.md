### Code for edge detection using pretrained hed model(caffe) using OpenCV

Command to run the edge detection model on video

    python edge.py --input video.mp4 --prototxt deploy.prototxt --caffemodel hed_pretrained_bsds.caffemodel 
    --width 300 --height 300

Command to run the edge detection model on image

    python edge_detector.py --input image.png --prototxt deploy.prototxt --caffemodel hed_pretrained_bsds.caffemodel
    --width 300 --height 300 
