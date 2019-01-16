import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageFilter

class Predict:
	model_path = None

	sess = None
	model = None

	#input
	# the shape of data input
	# more details in number_classify.py
	X = tf.placeholder(tf.float32,[None,28,28,1]) 

	#output
	Y = tf.placeholder(tf.float32,[None,10])

	def __init__(self,model_path=None):
		self.model_path = model_path

		self.init_session()
		self.load_model()

	def init_session(self):
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def rebuild_cnn_model(self):
		# Convolutional Layer 1

		conv1 = tf.layers.conv2d(inputs=self.X,
								filters=32,
								kernel_size=[5,5],
								padding="same",
								activation=tf.nn.relu)

		# Pooling 1

		pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2) # 14 x 14

		# Conv layer 2

		conv2 = tf.layers.conv2d(inputs=pool1,
								filters=64,
								kernel_size=[5,5],
								padding="same",
								activation=tf.nn.relu)

		# Pooling 2

		pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2) # 7x7


		# Conv layer 3

		conv3 = tf.layers.conv2d(inputs=pool2,filters=128,kernel_size=[5,5],padding="same",activation=tf.nn.relu)

		# Pooling 3

		pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=2) # 3x3x128

		# Dense 1

		pool3_flat = tf.reshape(pool3,[-1,3*3*128])

		dense = tf.layers.dense(inputs=pool3_flat,units=1024,activation=tf.nn.relu)

		dropout = tf.layers.dropout(inputs=dense,rate=0.4)

		# Logits Layer

		logits = tf.layers.dense(inputs=dropout,units=10)

		self.model = logits

		return self.model

	def load_model(self):
		self.rebuild_cnn_model()

		saver = tf.train.Saver()
		saver.restore(self.sess,self.model_path)

	def processImage(self,img):
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		im_pil = Image.fromarray(img)
		im_pil = im_pil.resize((28,28),Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

		data = [(255-x) * 1.0 / 255.0 for x in list(im_pil.getdata())]

		cv2.imshow('number',np.array(im_pil))

		data_reshaped = np.reshape(data,(-1,28,28,1)).tolist()

		return data_reshaped

	def predict(self,img):
		data = self.processImage(img)

		number = self.sess.run(tf.argmax(self.model,1),{self.X:data})[0]
		accuracy = self.sess.run(tf.nn.softmax(self.model,name="softmax_tensor"),{self.X:data})[0]

		return number,accuracy[number]