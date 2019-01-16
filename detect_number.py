import cv2
import numpy as np
from Predict import Predict
from predict_number import MNISTTester
from PIL import Image,ImageFilter

# define global variables

WINDOW_HEIGHT = 600
WINDOW_WIDTH = 600
BAR_HEIGHT = 100
BAR_COLOR = (100,76,60)

#for drawing
drawing_points = []
isDrawing = False
COLOR_CHANNEL = 3
LINE_COLOR = (0,0,0)
LINE_THICKNESS = 10
IS_CLOSED_SHAPE = False

# make white window
mainWindow = None

def InitMainWindow():
	createMainWindow()
	createEraseBar()
	createPredictionTable()

def createMainWindow():
	global mainWindow
	mainWindow = np.ones((WINDOW_WIDTH,WINDOW_HEIGHT,COLOR_CHANNEL),np.uint8) * 255

def createEraseBar():
	global mainWindow
	mainWindow[:100,:] = BAR_COLOR
	cv2.putText(mainWindow,'Erase',(WINDOW_WIDTH/2 - 20,BAR_HEIGHT/2),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2,cv2.LINE_AA)

def createPredictionTable():
	global mainWindow
	mainWindow[500:,:] = BAR_COLOR
	cv2.putText(mainWindow,'Predict:',(10,500 + BAR_HEIGHT/2),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(mainWindow,'Probability:',(350,500 + BAR_HEIGHT/2),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2,cv2.LINE_AA)

# Definition for drawing
def InitDrawingEvent():
	cv2.namedWindow('Number Detection')
	cv2.setMouseCallback('Number Detection',drawingCallBack)

def drawingCallBack(event,x,y,flags,param):
	global drawing_points
	global isDrawing
	global mainWindow

	if event == cv2.EVENT_LBUTTONDOWN:
		if(mouseInClearArea(y) == True):
			clearPaintingWindow()
		else:
			startDrawing()

	elif event == cv2.EVENT_MOUSEMOVE:
		drawingInProcess(x,y)

	elif event == cv2.EVENT_LBUTTONUP:
		finishDrawing()

def clearPaintingWindow():
	global mainWindow
	mainWindow[100:500,:] = 255

def mouseInClearArea(y):
	if (y <= 100):
		return True
	return False

def startDrawing():
	global isDrawing
	isDrawing = True

def drawingInProcess(x,y):
	global isDrawing
	global drawing_points

	if(isDrawing):
		if (y >= 100 and y <= 500):
			drawing_points.append((x,y))
			pts = np.array(drawing_points,np.int32)
		pts = pts.reshape((-1,1,2))
		cv2.polylines(mainWindow,[pts],IS_CLOSED_SHAPE,LINE_COLOR,LINE_THICKNESS)	

def finishDrawing():
	global isDrawing
	global drawing_points

	isDrawing = False
	drawing_points = []

def drawOutputToPredictionTable(number,accuracy):
	global mainWindow
	mainWindow[500:600,100:200] = BAR_COLOR
	mainWindow[500:600,500:600] = BAR_COLOR
	cv2.putText(mainWindow,str(number),(100,500 + BAR_HEIGHT/2),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(mainWindow,"{:.2f}%".format(accuracy),(500,500 + BAR_HEIGHT/2),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2,cv2.LINE_AA)

# definition for shape combining
def combineToOneBoundingRect(boundingRects):
	x1_arr = []
	y1_arr =[]
	x2_arr = []
	y2_arr = []

	for i in range(len(boundingRects)):
		x1_arr.append(boundingRects[i][0])
		y1_arr.append(boundingRects[i][1])
		x2_arr.append(boundingRects[i][2])
		y2_arr.append(boundingRects[i][3])

	x1 = np.min(np.array(x1_arr))
	y1 = np.min(np.array(y1_arr))
	x2 = np.max(np.array(x2_arr)) + x1
	y2 = np.max(np.array(y2_arr)) + y1

	return (x1,y1,x2,y2)


# definition for finding contours
def findContours(img):
	minThreshold = 100
	maxThreshold = 200
	canny = cv2.Canny(img,minThreshold,maxThreshold)

	cv2.imshow('canny',canny)

	contours,_ = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	contours_poly = [None]*len(contours)
	boundRect = [None]*len(contours)

	for i,c in enumerate(contours):
		contours_poly[i] = cv2.approxPolyDP(c,3,True)
		boundRect[i] = cv2.boundingRect(contours_poly[i])

	return boundRect,len(contours)

def drawBoundRect(img,boundRect,contours_len):

	for i in range(contours_len):
		x = int(boundRect[i][0])
		y = int(boundRect[i][1])
		width = x + int(boundRect[i][2])
		height = y + int(boundRect[i][3])

		cv2.rectangle(img,(x,y),(width,height),(255,0,255),2)

	return img

def fitdata(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	im_pil = Image.fromarray(img)
	im_pil = im_pil.resize((28,28),Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

	cv2.imshow('fit',np.array(im_pil))

	data = [(255-x) * 1.0 / 255.0 for x in list(im_pil.getdata())]

	return np.reshape(data,(-1,28,28,1)).tolist()


def main():
	InitMainWindow()
	InitDrawingEvent()

	#mnist = MNISTTester(model_path='/home/quangdinh/Practice/CNN/models/mnist-cnn',data_path=None)
	mnist_predict = Predict(model_path="/home/quangdinh/Practice/CNN/Trained/Mnist_Model_5/model.ckpt-20000")

	paintingFrame = mainWindow[100:500,:]

	while(True):
		cv2.imshow('Number Detection',mainWindow)

		boundRect,contours_len = findContours(paintingFrame)

		if len(boundRect) > 0:
			box = combineToOneBoundingRect(boundRect)

			(x1,y1,x2,y2) = box

			number_img = paintingFrame[y1-20:y2+20,x1-20:x2+20]

			#data = fitdata(number_img)

			#number,accuracy = mnist.classify({mnist.X:data})

			number,accuracy = mnist_predict.predict(number_img)

			#print("Number: {} | accuracy: {}".format(number,accuracy))

			drawOutputToPredictionTable(number,accuracy*100)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()

main()
