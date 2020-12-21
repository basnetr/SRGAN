import tensorflow as tf
import numpy as np
import imageio
import os

def get_test_lr():
    batch_size = 8
    lr_size = 32
    batch_data = np.zeros((batch_size,lr_size,lr_size,3))
    for i in range(8):
        filename = 'test_lr\\' + str(i+1) + '.png'
        batch_data[i,:,:,:] = (imageio.imread(filename,pilmode="RGB"))/255 # normalizing image 
    return np.array(batch_data).astype(np.float32)

def get_test_hr():
    batch_size = 8
    hr_size = 128
    batch_data = np.zeros((batch_size,hr_size,hr_size,3))
    for i in range(8):
        filename = 'test_hr\\' + str(i+1) + '.png'
        batch_data[i,:,:,:] = (imageio.imread(filename,pilmode="RGB")-127.5)/127.5 # normalizing image 
    return np.array(batch_data).astype(np.float32)

def do_test(itr):
	op_dir = 'test_op_' + str(itr) + '/'
	if not os.path.isdir(op_dir):
		os.mkdir(op_dir); print('Directory to save test created.')

	model_dir = 'srgan_models/' #the slash is important
	model_name = 'srgan_model'

	sess = tf.Session()
	savediter = str(itr)

	print('Importing meta graph...')
	saver = tf.train.import_meta_graph(model_dir + model_name + '-' + savediter + '.meta')

	print('Restoring model : ' + model_name + '-' + savediter)
	saver.restore(sess, model_dir + model_name + '-' + savediter)

	graph = tf.get_default_graph()
	input_hr = graph.get_tensor_by_name("hr_image_batch:0")
	input_lr = graph.get_tensor_by_name("lr_image_batch:0")
	training = graph.get_tensor_by_name("training:0")

	conv5 = graph.get_tensor_by_name("generator/conv5:0")
	conv5 = tf.tanh(conv5)

	feed_dict={input_hr: get_test_hr(), input_lr: get_test_lr(), training:False}

	[conv5] = sess.run([conv5], feed_dict=feed_dict)

	print('Saving Super-Resolved Images...')
	for i in range(8):
	    im = conv5[i,:,:,:]
	    im = (im+1)/2
	    imageio.imwrite(op_dir + str(i) + '.png',im)   

model_num = 30000
print('Initiated test for model - ' + str(model_num))
do_test(model_num)  