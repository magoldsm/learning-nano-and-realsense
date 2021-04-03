## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import jetson.inference
import jetson.utils
import time
# import codetiming
# import SysTimer as codetiming
import NoTimer as codetiming
import NMS

INCHES_PER_METER = 39.37
PADDING_FACTOR = 0.1
CONFIDENCE_THRESHOLD=0.19

HRES_RGB=640
VRES_RGB=480
HRES_DEPTH=640
VRES_DEPTH=480

# Given an object's bounding box (pt1, pt2) and the padding_factor, return the bounding
# box over which the NN has computed depth

def average_depth_coord(pt1, pt2, padding_factor):
    factor = 1 - padding_factor
    x_shift = (pt2[0] - pt1[0]) * factor / 2
    y_shift = (pt2[1] - pt1[1]) * factor / 2
    avg_pt1 = int(pt1[0] + x_shift), int(pt1[1] + y_shift)
    avg_pt2 = int(pt2[0] - x_shift), int(pt2[1] - y_shift)
    return avg_pt1, avg_pt2

with codetiming.Timer("Load network: {milliseconds:.1f} ms"):
    net = jetson.inference.detectNet(argv=['--model=cones-and-cells/ssd-mobilenet.onnx', '--labels=cones-and-cells/labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes', '--threshold={:.2f}'.format(CONFIDENCE_THRESHOLD)])

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
depth_sensor = device.first_depth_sensor()

DEPTH_SCALE = depth_sensor.get_depth_scale()

config.enable_stream(rs.stream.depth, HRES_DEPTH, VRES_DEPTH, rs.format.z16, 30)

# if device_product_line == 'L500':
#     config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
# else:
config.enable_stream(rs.stream.color, HRES_RGB, VRES_RGB, rs.format.bgr8, 30)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
# align_to = rs.stream.depth
align = rs.align(align_to)

cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
cv2.resizeWindow('RealSense', (HRES_RGB*2, VRES_RGB))

# Start streaming
pipeline.start(config)

last_frame_time = time.perf_counter()
lastFrameNumber = 0

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        # with codetiming.Timer(text="wait_for_frames: {milliseconds:.0f} ms"):
        unaligned_frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        # with codetiming.Timer(text="align: {milliseconds:.0f} ms"):
        frames = align.process(unaligned_frames)
        # frames = unaligned_frames

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        now = time.perf_counter()
        deltaT = now - last_frame_time
        last_frame_time = now

        rsFrameRate = 1 / deltaT

        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # Convert images to numpy arrays

        # with codetiming.Timer(text="numpy: {milliseconds:.0f} ms"):
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # with codetiming.Timer(text="applyColorMap: {milliseconds:.0f} ms"):
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # with codetiming.Timer(text="cudaFromNumpy: {milliseconds:.0f} ms"):
        imgCuda = jetson.utils.cudaFromNumpy(color_image)
        # with codetiming.Timer(text="Detect: {milliseconds:.0f} ms"):
        detections = net.Detect(imgCuda, depth_colormap_dim[1], depth_colormap_dim[0])
        # rawDetections = net.Detect(imgCuda, depth_colormap_dim[1], depth_colormap_dim[0])
        # detections = NMS.nms(rawDetections, 0.65)

        b,g,r = color_image[0, 0]
        if b == 123 and g == 123 and r == 123:
            print("scribbled frame")
        else:
            with codetiming.Timer(text="Annotate: {milliseconds:.0f} ms"):
                if detections:
                    # print("{} detections".format(len(detections)))
                    idxD = 0
                    for detection in detections:
                        # idxD += 1
                        # print("detection #{}".format(idxD))
                        with codetiming.Timer(text="detection: {milliseconds:.0f} ms"):
                            with codetiming.Timer(text="init: {milliseconds:.0f} ms"):
                                tl = int(detection.Left), int(detection.Top)
                                br = int(detection.Right), int(detection.Bottom)
                                ct = int(detection.Center[0]), int(detection.Center[1])
                                bl = int(detection.Left), int(detection.Bottom)
                                avg_tl, avg_br = average_depth_coord(tl, br, PADDING_FACTOR)
                                ptx = avg_tl[0], avg_br[1]+12
                                pty = avg_tl[0], avg_br[1]+24
                                ptz = avg_tl[0], avg_br[1]+36

                                depthSum = 0
                                depthCount = 0

                            with codetiming.Timer(text="get_distance: {milliseconds:.1f} ms"):
                                for x in range(avg_tl[0], avg_br[0]):
                                    for y in range(avg_tl[1], avg_br[1]):
                                        d = depth_frame.get_distance(x, y)
                                        if d != 0:
                                            depthSum += d
                                            depthCount += 1
                                if depthCount == 0:
                                    depth = 0
                                else:
                                    depth = depthSum / depthCount

                            # with codetiming.Timer(text="numpy: {milliseconds:.1f} ms"):
                            #     a = depth_image[avg_tl[1]:avg_br[1], avg_tl[0]:avg_br[0]]
                            #     d = np.sum(a) * DEPTH_SCALE / np.count_nonzero(a)


                            with codetiming.Timer(text="deproject: {milliseconds:.1f} ms"):
                                center_xyz = rs.rs2_deproject_pixel_to_point(depth_intrinsics, ct, depth_frame.get_distance(ct[0], ct[1]))

                            # t = center_xyz[0]* INCHES_PER_METER

                            with codetiming.Timer(text="drawing: {milliseconds:.1f} ms"):
                                cv2.rectangle(color_image, tl, br, (0, 255, 255), 1)
                                cv2.putText(color_image, "{}".format(detection.ClassID), tl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255))
                                cv2.putText(color_image, "{:.1f}%".format(detection.Confidence*100), bl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255))

                                cv2.rectangle(depth_colormap, avg_tl, avg_br, (0, 0, 0), 1)

                                cv2.putText(depth_colormap, "x: "+"{:.1f}".format(center_xyz[0]* INCHES_PER_METER), ptx, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
                                cv2.putText(depth_colormap, "y: "+"{:.1f}".format(center_xyz[1]* INCHES_PER_METER), pty, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
                                cv2.putText(depth_colormap, "z: "+"{:.1f}".format(depth* INCHES_PER_METER), ptz, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

        color_image[0, 0] = [123, 123, 123]

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        cv2.putText(images, "ObjectDetection | Network {:2.0f} FPS".format(net.GetNetworkFPS()), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        cv2.putText(images, "     RealSense | Overall   {:2.0f} FPS".format(rsFrameRate), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

        # Show images
        cv2.imshow('RealSense', images)

        # if len(detections) > 1:
        #     cv2.waitKey(0)

        if cv2.waitKey(1) == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()
    