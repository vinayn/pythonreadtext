import numpy as np
import cv2
import math
from imutils.object_detection import non_max_suppression
import pytesseract
import time

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

    #(all_corners, all_corners_confidences) = get_rects(scores, geometry)
    # Return detections and confidences
    return [detections, confidences]

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




        [boxes, confidences] = decode(scores, geometry, 0.7)
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
                #  Get the minimum value of y axis . If it is not there then add id
                if min(boxroi[:, 1]) in orderbyYaxisminvalue :
                    orderbyYaxisminvalue[min(boxroi[:, 1])].append(boxroi)
                else :
                    orderbyYaxisminvalue[min(boxroi[:, 1])] = []
                    orderbyYaxisminvalue[min(boxroi[:, 1])].append(boxroi)



            except Exception as e:
                #print(' error min(boxroi[:, 1]) ' + str(min(boxroi[:, 1])) + ' max(boxroi[:, 1]) ' + str(max(
                #    boxroi[:, 1])) + ' min(boxroi[:, 0]) ' + str(min(boxroi[:, 0])) + ' max(boxroi[:, 0]) ' + str(
                #   max(boxroi[:, 0])))
                print(str(e))



            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            for j in range(4):
                p1 = (int(vertices[j][0]), int(vertices[j][1]))
                p2 = (int(vertices[(j + 1) % 4][0]), int(vertices[(j + 1) % 4][1]))
                cv2.line(frame, p1, p2, (0, 255, 0), 2, cv2.LINE_AA);

        #sort dictionary by key value
        sorted_keys = sorted(orderbyYaxisminvalue.keys())
        orderbyYaxisminvalue = {key: orderbyYaxisminvalue[key] for key in sorted_keys}

        #orderbyYaxisminvalue = sorted(orderbyYaxisminvalue.items())

        # to avoid the  rotation of the book or display
        temp_lowest_height = 7
        #some negative value
        mergingkey_y_axis = -100
        keys_to_be_removed = []


        #merge by nearest height value of y axis
        if orderbyYaxisminvalue:
            try:
                # order by xaxis min value
                #orderbyYaxisminvalue = sorted(orderbyYaxisminvalue,key=lambda x:x[1])
                for temporderbyyaxiskey in orderbyYaxisminvalue:
                    if mergingkey_y_axis != -100 and temporderbyyaxiskey <= (mergingkey_y_axis + temp_lowest_height) and temporderbyyaxiskey > mergingkey_y_axis:
                        orderbyYaxisminvalue[mergingkey_y_axis].extend(orderbyYaxisminvalue[temporderbyyaxiskey])
                        # delete the appended dictinary key
                        # del orderbyYaxisminvalue[temporderbyxaxiskey]

                        # to remove runtime error
                        keys_to_be_removed.append(temporderbyyaxiskey)
                        continue


                    mergingkey_y_axis = temporderbyyaxiskey

            except Exception as e:
                print(str(e))

        """
        #merge by nearest height value of y axis
        if orderbyYaxisminvalue:
            try:
                # order by xaxis min value
                #orderbyYaxisminvalue = sorted(orderbyYaxisminvalue,key=lambda x:x[1])
                for temporderbyxaxiskey in orderbyYaxisminvalue:

                    if mergingkey_y_axis != -100:
                        if temporderbyxaxiskey <= (mergingkey_y_axis+temp_lowest_height) and temporderbyxaxiskey > mergingkey_y_axis:
                            orderbyYaxisminvalue[mergingkey_y_axis].extend(orderbyYaxisminvalue[temporderbyxaxiskey])
                            #delete the appended dictinary key
                            #del orderbyYaxisminvalue[temporderbyxaxiskey]

                            #to remove runtime error
                            keys_to_be_removed.append(temporderbyxaxiskey)
                            continue


                    for eachboxkey,eachboxvalue in enumerate(orderbyYaxisminvalue[temporderbyxaxiskey]):
                        lowest_x,lowest_y,lowest_w,lowest_height = cv2.boundingRect(orderbyYaxisminvalue[temporderbyxaxiskey][eachboxkey])

                        #get the lowest height
                        if temp_lowest_height == 0 :
                            temp_lowest_height = lowest_height
                        else :
                            if lowest_height < temp_lowest_height:
                                temp_lowest_height = (lowest_height/2)


                    if mergingkey_y_axis == -100:
                        mergingkey_y_axis = temporderbyxaxiskey


            except Exception as e:
                print(str(e))



        """

        for k in keys_to_be_removed:
            del orderbyYaxisminvalue[k]


        #print(" printing new detection ")
        # order by xaxis min value
        if orderbyYaxisminvalue :
            try:
                # order by xaxis min value
                #orderbyYaxisminvalue = sorted(orderbyYaxisminvalue,key=lambda x:x[1])

                #tempdicorder = dict(sorted(orderbyYaxisminvalue.items(), key=lambda item: item[1]))

                for temporderbyxaxiskey in orderbyYaxisminvalue:


                    """    
                    for eachboxkey,eachboxvalue in enumerate(orderbyYaxisminvalue[temporderbyxaxiskey]) :
                        orderbyYaxisminvalue[temporderbyxaxiskey][eachboxkey] = np.asarray(sorted(orderbyYaxisminvalue[temporderbyxaxiskey][eachboxkey],
                                                                           key=lambda x: x[0]))
                                                                           
                    """



                # Related text
                relatedtext = None
                for temporderbyxaxiskey in orderbyYaxisminvalue:
                    for eachboxvalue in orderbyYaxisminvalue[temporderbyxaxiskey]:
                        roi = frame[min(eachboxvalue[:, 1]):max(eachboxvalue[:, 1]), min(eachboxvalue[:, 0]):max(eachboxvalue[:, 0])]
                        config = ("-l eng --oem 1 --psm 7")
                        if relatedtext is None:
                            relatedtext =  pytesseract.image_to_string(roi, config=config).replace('\n', '').replace('\f', '')
                        else:
                            relatedtext += ' ' + pytesseract.image_to_string(roi, config=config).replace('\n', '').replace('\f', '')


                    print(relatedtext)
                    #empty the string
                    relatedtext = None


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