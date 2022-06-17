import numpy as np
import cv2
import os
import imutils
import time
import logging
from datetime import datetime


def yolo_detect(share_value):
	logger = logging.getLogger(__name__)

	# handler 생성 (stream, file)
	streamHandler = logging.StreamHandler()
	fileHandler = logging.FileHandler('./yolo/log/0617_witcam.log')

	# logger instance에 handler 설정
	logger.addHandler(streamHandler)
	logger.addHandler(fileHandler)
	logger.setLevel(level=logging.DEBUG)

	NMS_THRESHOLD = 0.3


	def draw_text(img, text, x, y):
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 1
		font_thickness = 2
		text_color = (255, 0, 0)
		text_color_bg = (0, 0, 0)

		text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
		text_w, text_h = text_size
		offset = 5

		cv2.rectangle(img, (x - offset, y - offset), (x + text_w + offset, y + text_h + offset), text_color_bg, -1)
		cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)


	def pedestrian_detection(image, model, layer_name, min_confidence, personidz=0):
		(H, W) = image.shape[:2]
		results = []

		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		model.setInput(blob)
		layerOutputs = model.forward(layer_name)

		boxes = []
		centroids = []
		confidences = []

		for output in layerOutputs:
			for detection in output:

				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				if classID == personidz and confidence > min_confidence:

					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boxes.append([x, y, int(width), int(height)])
					centroids.append((centerX, centerY))
					confidences.append(float(confidence))
		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idzs = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, NMS_THRESHOLD)
		# ensure at least one detection exists
		if len(idzs) > 0:
			# loop over the indexes we are keeping
			for i in idzs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				# update our results list to consist of the person
				# prediction probability, bounding box coordinates,
				# and the centroid
				res = (confidences[i], (x, y, x + w, y + h), centroids[i])
				results.append(res)
		# return the list of results
		return results


	labelsPath = "./yolo/coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	weights_path = "./yolo/yolov4-tiny.weights"
	config_path = "./yolo/yolov4-tiny.cfg"

	model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
	'''
	model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	'''

	layer_name = model.getLayerNames()
	layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

	# Connect to Webcam
	cap = cv2.VideoCapture(0)
	writer = None

	# Get FPS
	fps = cap.get(cv2.CAP_PROP_FPS)

	if fps == 0.0:
		fps = 30.0

	time_per_frame_video = 1 / fps
	last_time = time.perf_counter()

	MIN_CONFIDENCE = 0.7

	while True:
		# 현재 시간에 대해 사람이 감지됐는지 확인
		now = datetime.now()
		(grabbed, image) = cap.read()

		if not grabbed:
			break

		image = imutils.resize(image, width=700)

		# 밝기 어둡게
		image = cv2.subtract(image, (100, 100, 100, 0))

		thres = share_value.recv()

		if thres != -1:
			MIN_CONFIDENCE = thres
			print("===recv====")
			print(thres)
			print(MIN_CONFIDENCE)

		results = pedestrian_detection(image, model, layer_name, MIN_CONFIDENCE, personidz=LABELS.index("person"))

		# # 사람이 검출되는 경우와 아닌 경우 나눠서 logging
		if results:
			state = "{} Detect {}".format(now, MIN_CONFIDENCE)
			logger.info(state)
		else:
			state = "{} None {}".format(now, MIN_CONFIDENCE)
			logger.info(state)

		for res in results:
			cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)

		# fsp 계산
		time_per_frame = time.perf_counter() - last_time
		time_sleep_frame = max(0, time_per_frame_video - time_per_frame)
		time.sleep(time_sleep_frame)

		real_fps = 1 / (time.perf_counter() - last_time)
		last_time = time.perf_counter()

		x = 30
		y = 50
		text = '%.2f fps' % real_fps

		# 이미지의 (x, y)에 텍스트 출력
		draw_text(image, text, x, y)

		cv2.imshow("Detection", image)

		key = cv2.waitKey(1)
		if key == 27:
			break

	cap.release()
	cv2.destroyAllWindows()


