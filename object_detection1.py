import numpy as np 
import cv2

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

#class labels are in the yolov3.txt file
classes = []

with open ("yolov3.txt", "r") as f:
	#for each line in the file, we are going to strip the words and put it in our list
	classes = [line.strip() for line in f.readlines()]

#Getting the layers of the model
layer_names = net.getLayerNames()
outputLayer = [layer_names[i[0] -1] for i in net.getUnconnectedOutLayers()]


#Going to be using our camera
cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()

	width, height, channel = frame.shape
	

	#Need to convert image into a 'blob' first

	#image, scale factor, tweaking for YOLO and True to shift to RGB
	blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)

	#Done
	net.setInput(blob)
	outs = net.forward(outputLayer)

	for out in outs:
		for detection in out:
			#Will be working with the confidence
			scores = detection[5:]

			#class_id is the id for each of the classes in our classes[] array
			#The label (or class) predicted is the maximum score
			class_id = np.argmax(scores)

			#Confidence that the label is correct is stored at the same index in "scores" as in "classes[]"
			confidence = scores[class_id]

			if confidence > 0.5:
				centre_x = int(detection[0] * width)
				centre_y = int(detection[1]* height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				true_x = int(centre_x-w/2)
				true_y = int(centre_y-h/2)

				cv2.rectangle(frame, (true_x, true_y), (true_x+w, true_y+h), (0,0,255), 2)

				#class_id is the id of the prediction (key to the actual string in classes[])
				label = str(classes[class_id])
				cv2.putText(frame, label, (true_x, true_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
				



	cv2.imshow("frame", frame)

	if cv2.waitKey(2) & 0XFF == ord('x'):
		break

cap.release()
cv2.destroyAllWindows()