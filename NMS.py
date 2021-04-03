def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    if iou > .02:
    
    
        print("iou(({:.0f}, {:.0f}, {:.0f}, {:.0f}) and ({:.0f}, {:.0f}, {:.0f}, {:.0f})) = {:.2f}".format(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], iou))

    return iou



def nms(detections, threshold):
    final = []
    nDet = len(detections)
    bRemoved = False

    if nDet >= 3:
        print("here")

    while len(detections) > 0:
        maxC = 0
        maxD = None
        for detection in detections:
            if detection.Confidence > maxC:
                maxC = detection.Confidence
                maxD = detection
        final.append(maxD)
        detections.remove(maxD)

        for i in range(len(detections) - 1, -1, -1):
            detection = detections[i]
            IoU = get_iou([maxD.Left, maxD.Top, maxD.Right, maxD.Bottom], [detection.Left, detection.Top, detection.Right, detection.Bottom])
            if IoU > threshold:
                bRemoved = True
                detections.remove(detection)

    if bRemoved:
        print("Started with {} and ended with {}".format(nDet, len(final)))

    return final
