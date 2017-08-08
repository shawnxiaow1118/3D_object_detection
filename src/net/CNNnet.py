import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]

class rgb_net:
	def __init__(self, var_file=None):
		if (var_file is None):
			# save_path
			pass
		else:
			# self.dict = np.load(var_file).item()
			pass

		self.imgs = imgs

	def inference():
		# zero mean input
		mean = tf.constant(VGG_MEAN, dtype=tf.float32, shape=[1,1,1,3], name='img_mean')
		imgs = self.imgs - mean

		# color transform
		with tf.name_scope('color_space') as scope:
			kernel = tf.Variable(tf.truncated_normal([1,1,1,3]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[3],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv = tf.nn.relu(out, name=scope)


		with tf.name_scope('conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,3,64]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv1_1 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv1_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,64,64]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv1_2 = tf.nn.relu(out, name=scope)

		with tf.name_scope('pool_1') as scope:
			pool_1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
									padding='SAME',	name='pool1')

		with tf.name_scope('conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,64,128]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[128],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_1 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv2_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,128,128]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[128],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_2 = tf.nn.relu(out, name=scope)

		with tf.name_scope('pool_2') as scope:
			pool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
									padding='SAME',	name='pool2')

		with tf.name_scope('conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,128,256]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_1 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv3_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,256,256]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv3_2 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv3_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,256,256]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv3_3 = tf.nn.relu(out, name=scope)

		with tf.name_scope('pool_3') as scope:
			pool_4 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
									padding='SAME',	name='pool3')

		with tf.name_scope('conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,256,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_1 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv4_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_2 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv4_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_1 = tf.nn.relu(out, name=scope)


		with tf.name_scope('pool_4') as scope:
			pool_4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
									padding='SAME',	name='pool4')

		with tf.name_scope('conv5_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv5_1 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv5_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv5_2 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv5_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv5_3 = tf.nn.relu(out, name=scope)

		with tf.name_scope('pool_5') as scope:
			pool_5 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
									padding='SAME',	name='pool5')
		return pool5


class front_net():
		def __init__(self, var_file=None):
		if (var_file is None):
			# save_path
			pass
		else:
			# self.dict = np.load(var_file).item()
			pass

		self.imgs = imgs

	def inference():
		# zero mean input
		mean = tf.constant(VGG_MEAN, dtype=tf.float32, shape=[1,1,1,3], name='img_mean')
		imgs = self.imgs - mean

		# color transform
		with tf.name_scope('color_space') as scope:
			kernel = tf.Variable(tf.truncated_normal([1,1,1,3]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[3],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv = tf.nn.relu(out, name=scope)


		with tf.name_scope('conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,3,64]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv1_1 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv1_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,64,64]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv1_2 = tf.nn.relu(out, name=scope)

		with tf.name_scope('pool_1') as scope:
			pool_1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
									padding='SAME',	name='pool1')

		with tf.name_scope('conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,64,128]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[128],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_1 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv2_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,128,128]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[128],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_2 = tf.nn.relu(out, name=scope)

		with tf.name_scope('pool_2') as scope:
			pool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
									padding='SAME',	name='pool2')

		with tf.name_scope('conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,128,256]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_1 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv3_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,256,256]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv3_2 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv3_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,256,256]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv3_3 = tf.nn.relu(out, name=scope)

		with tf.name_scope('pool_3') as scope:
			pool_4 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
									padding='SAME',	name='pool3')

		with tf.name_scope('conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,256,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_1 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv4_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_2 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv4_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv2_1 = tf.nn.relu(out, name=scope)


		with tf.name_scope('pool_4') as scope:
			pool_4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
									padding='SAME',	name='pool4')

		with tf.name_scope('conv5_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv5_1 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv5_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv5_2 = tf.nn.relu(out, name=scope)

		with tf.name_scope('conv5_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32, stddev=1e-1, name='weights')
			conv = tf.conv2d(imgs, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32), trainable=True,name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv5_3 = tf.nn.relu(out, name=scope)

		with tf.name_scope('pool_5') as scope:
			pool_5 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
									padding='SAME',	name='pool5')
		return pool5
