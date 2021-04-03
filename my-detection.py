import jetson.inference
import jetson.utils

# net = jetson.inference.detectNet(argv=['--model=Cones-and-Cells/ssd-mobilenet.onnx --input-blob=input_0 --output-cvg=scores --output-bbox=boxes --labels=Cones-and-Cells/labels.txt'])
# net = jetson.inference.detectNet(argv=['--model=Cones-and-Cells/ssd-mobilenet.onnx --labels=Cones-and-Cells/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes'])
net = jetson.inference.detectNet(argv=['--model=cones-and-cells/ssd-mobilenet.onnx', '--labels=cones-and-cells/labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes', '--threshold=0.18'])
# net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(640, 360, "/dev/video3")
# camera = jetson.utils.gstCamera(1280, 720, "/dev/video3")
# camera = jetson.utils.videoSource("/dev/video4")      # '/dev/video0' for V4L2
# display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
display = jetson.utils.glDisplay()

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()
    detections = net.Detect(img, width, height)
    display.RenderOnce(img, width, height)
    display.SetTitle("ObjectDetection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    

# while display.IsStreaming():
	# img = camera.Capture()
	# detections = net.Detect(img)
	# display.Render(img)
	# display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS())
