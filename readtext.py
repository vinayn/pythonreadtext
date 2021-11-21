import numpy as np
import cv2
import math
from imutils.object_detection import non_max_suppression
import pytesseract
import time

def get_rects(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it

			#if scoresData[x] < 0.5:
			#	continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if (score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = (
            [offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    (all_corners, all_corners_confidences) = get_rects(scores, geometry)
    # Return detections and confidences
    return [detections, confidences, all_corners, all_corners_confidences]

cap = cv2.VideoCapture(0)

framevideoH = 320
framevideoW = 320

cap.set(cv2.CAP_PROP_FRAME_WIDTH, framevideoW);
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, framevideoH);

# Capture Frames
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (framevideoW,framevideoH))

#There are two outputs of the network. One specifies the geometry of the Text-box and the other specifies the confidence score of the detected box
outputLayers = []
outputLayers.append("feature_fusion/Conv_7/Sigmoid")
outputLayers.append("feature_fusion/concat_3")

net = cv2.dnn.readNet('frozen_east_text_detection.pb')

while(cap.isOpened()):
    ret, frame = cap.read()

    #resize is happening here. Video resize not working
    frame = cv2.resize(frame, (framevideoW, framevideoH))

    if ret==True:
        #frame = cv2.flip(frame,0)

        # resize the frame, maintaining the aspect ratio
        #frame = imutils.resize(frame, width=1000)
        orig = frame.copy()

        (W, H) = (None, None)
        (newW, newH) = (framevideoW, framevideoH)
        (rW, rH) = (None, None)

        if W is None or H is None:
            (H, W) = frame.shape[:2]
            rW = W / float(newW)
            rH = H / float(newH)
        # resize the frame, this time ignoring aspect ratio
        frame = cv2.resize(frame, (newW, newH))

        #add additional resize
        #out.write(frame)

        blob = cv2.dnn.blobFromImage(frame, 1.0, (framevideoW, framevideoH), (123.68, 116.78, 103.94), True, False)

        net.setInput(blob)
        output = net.forward(outputLayers)

        scores = output[0]
        geometry = output[1]




        [boxes, confidences, all_corners, all_corners_confidences] = decode(scores, geometry, 0.7)
        # Apply NMS
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.7, 0.4)

        # initialize the list of results
        results = []

        # initialize the array to show text
        orderbyYaxisminvalue = {}

        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])

            boxroi = np.int0(cv2.boxPoints(boxes[i[0]]))

            startX = int(min(boxroi[:, 0]) * rW)
            endX = int(max(boxroi[:, 0]) * rW)

            startY = int(min(boxroi[:, 1]) * rH)
            endY = int(max(boxroi[:, 1]) * rH)



            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            addpadding = 0.05

            dX = int((endX - startX) * addpadding)
            dY = int((endY - startY) * addpadding)

            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(W, endX + (dX * 2))
            endY = min(H, endY + (dY * 2))




            roi = frame[min(boxroi[:, 1]):max(boxroi[:, 1]), min(boxroi[:, 0]):max(boxroi[:, 0])]

            #print(' min(boxroi[:, 1]) ' + str(min(boxroi[:, 1])) + ' max(boxroi[:, 1]) ' + str(max(
            #    boxroi[:, 1])) + ' min(boxroi[:, 0]) ' + str(min(boxroi[:, 0])) + ' max(boxroi[:, 0]) ' + str(max(boxroi[:, 0])))

            # r = frame[startY:endY, startX:endX]
            config = ("-l eng --oem 1 --psm 7")
            try:

                #text = pytesseract.image_to_string(roi, config=config)
                #print(str(min(boxroi[:, 1])))
                # order by y axis minimum value
                if min(boxroi[:, 1]) in orderbyYaxisminvalue :
                    orderbyYaxisminvalue[min(boxroi[:, 1])].append(boxroi)
                else :
                    orderbyYaxisminvalue[min(boxroi[:, 1])] = []
                    orderbyYaxisminvalue[min(boxroi[:, 1])].append(boxroi)

                orderbyYaxisminvalue[min(boxroi[:, 1])].append(boxroi)
                #print(text)

            except Exception as e:
                #print(' error min(boxroi[:, 1]) ' + str(min(boxroi[:, 1])) + ' max(boxroi[:, 1]) ' + str(max(
                #    boxroi[:, 1])) + ' min(boxroi[:, 0]) ' + str(min(boxroi[:, 0])) + ' max(boxroi[:, 0]) ' + str(
                #   max(boxroi[:, 0])))
                print(str(e))
            #print('without padding')




            #roi = frame[startY:endY, startX:endX]
            #config = ("-l eng --oem 1 --psm 7")
            #text = pytesseract.image_to_string(roi, config=config)
           # print('with padding')
            #print(text)


            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            for j in range(4):
                p1 = (int(vertices[j][0]), int(vertices[j][1]))
                p2 = (int(vertices[(j + 1) % 4][0]), int(vertices[(j + 1) % 4][1]))
                cv2.line(frame, p1, p2, (0, 255, 0), 2, cv2.LINE_AA);

                # extract the region of interest
                # r = frame[startY:endY, startX:endX]
                #r = frame[vertices[j][1]:vertices[(j + 1) % 4][1], vertices[j][0]:vertices[(j + 1) % 4][0]]
                #r = frame[all_corners['startY']:all_corners['endY'], all_corners['startX']:all_corners['endX']]

                boxes_all_corners = non_max_suppression(np.array(all_corners), probs=all_corners_confidences)
                # loop over the bounding boxes
                for (startX, startY, endX, endY) in boxes_all_corners:
                    # scale the bounding box coordinates based on the respective
                    # ratios
                    startX = int(startX * rW)
                    startY = int(startY * rH)
                    endX = int(endX * rW)
                    endY = int(endY * rH)

                    # in order to obtain a better OCR of the text we can potentially
                    # apply a bit of padding surrounding the bounding box -- here we
                    # are computing the deltas in both the x and y directions
                    dX = int((endX - startX) * 0.0)
                    dY = int((endY - startY) * 0.0)

                    # apply padding to each side of the bounding box, respectively
                    startX = max(0, startX - dX)
                    startY = max(0, startY - dY)
                    endX = min(W, endX + (dX * 2))
                    endY = min(H, endY + (dY * 2))

                    # draw the bounding box on the frame
                    #cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)


                    # extract the actual padded ROI
                    #roi = frame[startY:endY, startX:endX]

                    # in order to apply Tesseract v4 to OCR text we must supply
                    # (1) a language, (2) an OEM flag of 4, indicating that the we
                    # wish to use the LSTM neural net model for OCR, and finally
                    # (3) an OEM value, in this case, 7 which implies that we are
                    # treating the ROI as a single line of text

                    #pytesseract.pytesseract.tesseract_cmd = 'D:/opencv/tessaractexe/vcpkg/installed/x64-windows-static/tools/tesseract/tesseract.exe'
                    #pytesseract.pytesseract.tesseract_cmd = 'C:/Users/nvina/.conda/envs/tesseract/Library/bin/tesseract.exe'

                    #config = ("-l eng --oem 1 --psm 7")
                    #text = pytesseract.image_to_string(roi, config=config)


                    # append bbox coordinate and associated text to the list of results
                    #results.append(((vertices[j][0], vertices[j][1], vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1]), text))
                    #results.append(((startX, startY, endX, endY), text))

                    #if len(text) > 4:
                    #    print(text)

                    # cv.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)

        if orderbyYaxisminvalue :
            try:
                # order by xaxis min value
                #orderbyYaxisminvalue = sorted(orderbyYaxisminvalue,key=lambda x:x[1])
                for temporderbyxaxiskey in orderbyYaxisminvalue:
                    #orderbyYaxisminvalue[temporderbyxaxiskey] = sorted(enumerate(orderbyYaxisminvalue[temporderbyxaxiskey]),key=lambda x: x[0])
                    for eachboxkey,eachboxvalue in enumerate(orderbyYaxisminvalue[temporderbyxaxiskey]) :
                        orderbyYaxisminvalue[temporderbyxaxiskey][eachboxkey] = np.asarray(sorted(orderbyYaxisminvalue[temporderbyxaxiskey][eachboxkey],
                                                                           key=lambda x: x[0]))
                    #print(' ok ' + str(temporderbyxaxiskey) + ' ok')
                    #print(orderbyYaxisminvalue[temporderbyxaxiskey])
                    #orderbyYaxisminvalue[temporderbyxaxiskey] = sorted(orderbyYaxisminvalue[temporderbyxaxiskey],key=lambda x:x[1][0])

                relatedtext = ''
                for temporderbyxaxiskey in orderbyYaxisminvalue:
                    for eachboxvalue in orderbyYaxisminvalue[temporderbyxaxiskey]:
                        roi = frame[min(eachboxvalue[:, 1]):max(eachboxvalue[:, 1]), min(eachboxvalue[:, 0]):max(eachboxvalue[:, 0])]
                        config = ("-l eng --oem 1 --psm 7")
                        relatedtext +=  pytesseract.image_to_string(roi, config=config)
                    print(relatedtext)
                    print('New line')

            except Exception as e:
                print(str(e))
        #empty the dictonary




        orderbyYaxisminvalue = {}
        #orderbyYaxisminvalue.sort(key=lambda x:x[1])
        #out.write(frame)
        # show the output frame
        cv2.imshow("Text Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()