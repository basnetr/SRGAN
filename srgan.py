import tensorflow as tf
import numpy as np
import imageio
import os

def get_batch_lr(datn, bn):
    batch_size = 8
    lr_size = 32
    batch_data = np.zeros((batch_size,lr_size,lr_size,3))
    batch = datn[(bn-1)*batch_size:bn*batch_size]
    for i in batch:
        filename = 'train_lr\\' + str(i) + '.png'
        batch_data[(i-1)%batch_size,:,:,:] = (imageio.imread(filename,pilmode="RGB"))/255 # normalizing image 
    return np.array(batch_data).astype(np.float32)

def get_batch_hr(datn, bn):
    batch_size = 8
    hr_size = 128
    batch_data = np.zeros((batch_size,hr_size,hr_size,3))
    batch = datn[(bn-1)*batch_size:bn*batch_size]
    for i in batch:
        filename = 'train_hr\\' + str(i) + '.png'
        batch_data[(i-1)%batch_size,:,:,:] = (imageio.imread(filename,pilmode="RGB")-127.5)/127.5 # normalizing image 
    return np.array(batch_data).astype(np.float32)

def get_val_lr():
    batch_size = 8
    lr_size = 32
    batch_data = np.zeros((batch_size,lr_size,lr_size,3))
    for i in range(8):
        filename = 'val_lr\\' + str(i+1) + '.png'
        batch_data[i,:,:,:] = (imageio.imread(filename,pilmode="RGB"))/255 # normalizing image 
    return np.array(batch_data).astype(np.float32)

def get_val_hr():
    batch_size = 8
    hr_size = 128
    batch_data = np.zeros((batch_size,hr_size,hr_size,3))
    for i in range(8):
        filename = 'val_hr\\' + str(i+1) + '.png'
        batch_data[i,:,:,:] = (imageio.imread(filename,pilmode="RGB")-127.5)/127.5 # normalizing image 
    return np.array(batch_data).astype(np.float32)


def generator(input_batch, is_train=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.truncated_normal_initializer(stddev=0.01)

        #conv_layer1 k9n64s1
        kernel1 = tf.get_variable('kernel1', [9,9,64,3],initializer=w_init, trainable=True)
        conv1 = tf.nn.conv2d_transpose(input_batch, filter=kernel1, output_shape=[8,32,32,64], strides=[1,1,1,1], padding='SAME', name='conv1')
        conv1 = tf.nn.relu(conv1)

        before_res = conv1 

        res1 =  res_block(conv1, is_train=True, scope='res1')
        res2 =  res_block(res1, is_train=True, scope='res2')
        res3 =  res_block(res2, is_train=True, scope='res3')
        res4 =  res_block(res3, is_train=True, scope='res4')
        res5 =  res_block(res4, is_train=True, scope='res5')

        #conv_layer2 k3n64s1
        kernel2 = tf.get_variable('kernel2', [3,3,64,64],initializer=w_init)
        conv2 = tf.nn.conv2d_transpose(res5, filter=kernel2, output_shape=[8,32,32,64], strides=[1,1,1,1], padding='SAME', name='conv2')
        bn1 = batch_normalize(conv2, training=is_train, name='conv2_bn')
        conv2 = tf.add(before_res, bn1)

        #conv_layer3 k3n256s1
        kernel3 = tf.get_variable('kernel3',[3,3,256,64],initializer=w_init)
        conv3 = tf.nn.conv2d_transpose(conv2, filter=kernel3, output_shape=[8,32,32,256], strides=[1,1,1,1], padding='SAME', name='conv3')
        conv3 = tf.nn.relu(pixel_shuffle_layer(conv3, 2, 64))

        #conv_layer4 k3n256s1
        kernel4 = tf.get_variable('kernel4', [3,3,256,64],initializer=w_init)
        conv4 = tf.nn.conv2d_transpose(conv3, filter=kernel4, output_shape=[8,64,64,256], strides=[1,1,1,1], padding='SAME', name='conv4')
        conv4 = tf.nn.relu(pixel_shuffle_layer(conv4, 2, 64))

        #conv_layer5 k9n3s1
        kernel5 = tf.get_variable('kernel5', [9,9,3,64],initializer=w_init)
        conv5 = tf.nn.conv2d_transpose(conv4, filter=kernel5, output_shape=[8,128,128,3], strides=[1,1,1,1], padding='SAME', name='conv5')


        print('<<< ---------- GENERATOR ----------- >>>')
        print(input_batch);	print('input_batch shape: ');	print(input_batch.shape)
        print(conv1);		print('conv1 shape: '); 		print(conv1.shape)
        print(res1);		print('res1 shape: '); 		    print(res1.shape)
        print(res2);		print('res2 shape: '); 		    print(res2.shape)
        print(res3); 		print('res3 shape: '); 		    print(res3.shape)
        print(res4); 		print('res4 shape: '); 		    print(res4.shape)
        print(res5); 		print('res5 shape: '); 		    print(res5.shape)
        print(conv2); 		print('conv2 shape: '); 		print(conv2.shape)
        print(conv3); 		print('conv3 shape: '); 		print(conv3.shape)
        print(conv4); 		print('conv4 shape: '); 		print(conv4.shape)
        print(conv5); 		print('conv5 shape: '); 		print(conv5.shape)


        return tf.tanh(conv5)

def res_block(input_batch, is_train=True, scope='res_block'):
	with tf.variable_scope(scope):
		w_init = tf.truncated_normal_initializer(stddev=0.1)

		start_block = input_batch

		kernel1 = tf.get_variable('kernel1', [3,3,64,64],initializer=w_init, trainable=True)
		conv1 = tf.nn.conv2d_transpose(input_batch, filter=kernel1, output_shape=[8,32,32,64], strides=[1,1,1,1], padding='SAME')

		#batch normalization
		bn1 = batch_normalize(conv1, training=is_train, name='bn1')
		bn1 = tf.nn.relu(bn1)

		kernel2 = tf.get_variable('kernel2', [3,3,64,64],initializer=w_init, trainable=True)
		conv2 = tf.nn.conv2d_transpose(bn1, filter=kernel2, output_shape=[8,32,32,64], strides=[1,1,1,1], padding='SAME')

		#batch normalization
		bn2 = batch_normalize(conv2, training=is_train, name='bn2')

		return tf.add(start_block, bn2) 

def batch_normalize(x, training, decay=0.99, epsilon=0.001, trainable=True, name='bn'):
    with tf.variable_scope(name):
        def bn_train():
            batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
            train_mean = tf.assign(
                pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(
                pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(
                    x, batch_mean, batch_var, beta, scale, epsilon)

        def bn_inference():
            return tf.nn.batch_normalization(
                x, pop_mean, pop_var, beta, scale, epsilon)

        dim = x.get_shape().as_list()[-1]
        beta = tf.get_variable(
            name='beta',
            shape=[dim],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.0),
            trainable=trainable)
        scale = tf.get_variable(
            name='scale',
            shape=[dim],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            trainable=trainable)
        pop_mean = tf.get_variable(
            name='pop_mean',
            shape=[dim],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),
            trainable=False)
        pop_var = tf.get_variable(
            name='pop_var', 
            shape=[dim],
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.0),
            trainable=False)
    return tf.cond(tf.equal(training, tf.constant(True)), bn_train, bn_inference)

def pixel_shuffle_layer(x, r, n_split):
    def PS(x, r):
        bs, a, b, c = x.get_shape().as_list()
        x = tf.reshape(x, (bs, a, b, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(x, a, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.split(x, b, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        return tf.reshape(x, (bs, a*r, b*r, 1))

    xc = tf.split(x, n_split, 3)
    return tf.concat([PS(x_, r) for x_ in xc], 3)

def lrelu(x, trainbable=None):
    alpha = 0.2
    return tf.maximum(alpha * x, x)

def discriminator(input_batch, is_train=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.truncated_normal_initializer(stddev=0.1)
        b_init = tf.constant_initializer(0.)

        #conv_layer1 k9n64s1
        kernel1 = tf.get_variable('kernel1', [3,3,3,64],initializer=w_init, trainable=True)
        conv1 = lrelu(tf.nn.conv2d(input_batch, filter=kernel1, strides=[1,1,1,1], padding='SAME', name='conv1'))

        kernel1_ = tf.get_variable('kernel1_', [3,3,64,64],initializer=w_init, trainable=True)
        conv1_ = tf.nn.conv2d(conv1, filter=kernel1_, strides=[1,2,2,1], padding='SAME', name='conv1_')
        conv1_ = lrelu(batch_normalize(conv1_, training=is_train, name='bn_conv1_'))

        kernel2 = tf.get_variable('kernel2', [3,3,64,128],initializer=w_init, trainable=True)
        conv2 = tf.nn.conv2d(conv1_, filter=kernel2, strides=[1,1,1,1], padding='SAME', name='conv2')
        conv2 = lrelu(batch_normalize(conv2, training=is_train, name='bn_conv2'))

        kernel2_ = tf.get_variable('kernel2_', [3,3,128,128],initializer=w_init, trainable=True)
        conv2_ = tf.nn.conv2d(conv2, filter=kernel2_, strides=[1,2,2,1], padding='SAME', name='conv2_')
        conv2_ = lrelu(batch_normalize(conv2_, training=is_train, name='bn_conv2_'))

        kernel3 = tf.get_variable('kernel3', [3,3,128,256],initializer=w_init, trainable=True)
        conv3 = tf.nn.conv2d(conv2_, filter=kernel3, strides=[1,1,1,1], padding='SAME', name='conv3')
        conv3 = lrelu(batch_normalize(conv3, training=is_train, name='bn_conv3'))

        kernel3_ = tf.get_variable('kernel3_', [3,3,256,256],initializer=w_init, trainable=True)
        conv3_ = tf.nn.conv2d(conv3, filter=kernel3_, strides=[1,2,2,1], padding='SAME', name='conv3_')
        conv3_ = lrelu(batch_normalize(conv3_, training=is_train, name='bn_conv3_'))

        kernel4 = tf.get_variable('kernel4', [3,3,256,512],initializer=w_init, trainable=True)
        conv4 = tf.nn.conv2d(conv3_, filter=kernel4, strides=[1,1,1,1], padding='SAME', name='conv4')
        conv4 = lrelu(batch_normalize(conv4, training=is_train, name='bn_conv4'))

        kernel4_ = tf.get_variable('kernel4_', [3,3,512,512],initializer=w_init, trainable=True)
        conv4_ = tf.nn.conv2d(conv4, filter=kernel4_, strides=[1,2,2,1], padding='SAME', name='conv4_')
        conv4_ = lrelu(batch_normalize(conv4_, training=is_train, name='bn_conv4_'))

        dim = conv4_.get_shape().as_list()
        conv4_flat = tf.reshape(conv4_, [-1, dim[1] * dim[2] * dim[3]])

        w_fc1 = tf.get_variable('w_fc1', shape=[conv4_flat.shape[-1], 1024], initializer=w_init, trainable=True)
        b_fc1 = tf.get_variable('b_fc1', [1024], initializer=b_init, trainable=True)

        fc1 = lrelu(tf.add(tf.matmul(conv4_flat, w_fc1), b_fc1))

        w_fc2 = tf.get_variable('w_fc2', shape=[1024, 1], initializer=w_init)
        b_fc2 = tf.get_variable('b_fc2', [1], initializer=b_init)

        fc2_logits = tf.add(tf.matmul(fc1, w_fc2), b_fc2)
        fc2_sigmoids = tf.sigmoid(fc2_logits)

        print('<<< ---------- DISCRIMINATOR ----------- >>>')
        print(input_batch);     print('input_batch shape: ');       print(input_batch.shape)
        print(conv1);           print('conv1 shape: ');             print(conv1.shape)
        print(conv1_);          print('conv1_ shape: ');            print(conv1_.shape)
        print(conv2);           print('conv2 shape: ');             print(conv2.shape)
        print(conv2_);          print('conv2_ shape: ');            print(conv2_.shape)
        print(conv3);           print('conv3 shape: ');             print(conv3.shape)
        print(conv3_);          print('conv3_ shape: ');            print(conv3_.shape)
        print(conv4);           print('conv4 shape: ');             print(conv4.shape)
        print(conv4_);          print('conv4_ shape: ');            print(conv4_.shape)
        print(conv4_flat);      print('conv4_flat shape: ');        print(conv4_flat.shape)
        print(fc1);             print('fc1 shape: ');               print(fc1.shape)
        print(fc2_logits);      print('fc2_logits shape: ');        print(fc2_logits.shape)

        return [fc2_logits, fc2_sigmoids]
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

def do_test(sess,itr,input_hr, input_lr, gen_hr):
    op_dir = 'test_op_' + str(itr) + '/'
    if not os.path.isdir(op_dir):
        os.mkdir(op_dir); print('Directory to save test created.')

    feed_dict={input_hr: get_test_hr(), input_lr: get_test_lr(), training:False}

    [gen_op] = sess.run([gen_hr], feed_dict=feed_dict)

    for i in range(8):
        im = gen_op[i,:,:,:]
        im = (im+1)/2
        imageio.imwrite(op_dir + str(i) + '.png',im) 

def PSNR(prediction, target): #only used for evaluation, not a part of loss function
    err_norm = target-prediction;
    err = (err_norm*127.5) + 127.5 #unnormalizing the error #size = batch_size x 128 x 128 x 3
    mse = tf.reduce_mean(tf.square(err),axis=(-3,-2,-1)) #calculating mse on each image : op = batch_size x 1
    psnr = 10 * tf.log(255*255/tf.sqrt(mse))  / np.log(10) #batch_size x 1
    avg_psnr = tf.reduce_mean(psnr)
    return avg_psnr

batch_size = 8;
hr_size = 128;
lr_size = 32;
nchannels = 3;
lr = 0.0001 #learning_rate

input_hr = tf.placeholder(tf.float32, [batch_size, hr_size, hr_size, nchannels], name="hr_image_batch")
input_lr = tf.placeholder(tf.float32, [batch_size, lr_size, lr_size, nchannels], name="lr_image_batch")
training = tf.placeholder(tf.bool, name="training")

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

gen_hr = generator(input_lr, is_train=training, reuse=False)
gen_psnr = PSNR(gen_hr, input_hr); print('<<< -- PSNR:  ', gen_psnr, ' -- >>>')

content_loss = tf.reduce_mean(tf.square(input_hr-gen_hr)) #we consider mse content loss

srres_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999).minimize(content_loss, global_step=global_step)

[real_logits, real_sigmoids] = discriminator(input_hr, is_train=training, reuse=False) #output of discriminator for input_hr, expected to be 1
[fake_logits, fake_sigmoids] = discriminator(gen_hr, is_train=training, reuse=True) #output of discriminator for generated_hr, expected to be 0

dloss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_sigmoids)) #labels = 1, all ones for original image #size = batch_size x 1
dloss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_sigmoids)) #labels = 0, all zeros for generated image  #size = batch_size x 1

dloss = tf.add(tf.reduce_mean(dloss_real),tf.reduce_mean(dloss_fake))*0.001; print('<<< -- dloss:  ', dloss, ' -- >>>')

gloss =  tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_sigmoids)) #labels = 1, for all generated image, gen's aim is to fool the dis

adversarial_loss = 0.001 * tf.reduce_mean(gloss)
perceptual_loss = content_loss + adversarial_loss; print('<<< -- perceptual_loss:  ', perceptual_loss, ' -- >>>')

d_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999).minimize(dloss)
g_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999).minimize(perceptual_loss, global_step=global_step)

#Directory to save models
model_dir = 'srgan_models/' #the slash is important
model_name = 'srgan_model'
load_model = 0 #Enter the previously saved model number

if not os.path.isdir(model_dir):
    os.mkdir(model_dir); print('Directory to save model created.')
    
#now training
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)

        #load model
        if load_model > 0:
            print('Loading Model... : ' + str(load_model))
            saver = tf.train.import_meta_graph(model_dir + model_name + '-' + str(load_model) + '.meta')
            saver.restore(sess, model_dir + model_name + '-' + str(load_model))

        train_size = 3137
        datn = list(range(1,train_size+1))

        # Training
        batch_size = 8
        n_batch = int(len(datn)/batch_size)
        n_epoch = 50000
        count = load_model;
        lr = 0.0001;

        dloss_arr = []
        gloss_arr = []
        vdloss_arr = []
        vgloss_arr = []

        closs_arr = []

        for _ in range(n_epoch):
            for itr in range(1,n_batch+1):
                count = count + 1

                batch_lr = get_batch_lr(datn,itr)
                batch_hr = get_batch_hr(datn,itr)

                if count < 20000: # 20000
                    closs,spsnr,_ = sess.run([content_loss, gen_psnr, srres_opt], feed_dict={input_hr: batch_hr, input_lr: batch_lr, training:True})
                    closs_arr.append(closs)
                    print('Loss ', count, '--', global_step.eval(), '--', ' | closs --> ', closs, ' | PSNR -->', spsnr)
                else:
                    feed_dict={input_hr: batch_hr, input_lr: batch_lr, training:True}
                    d_loss,_ = sess.run([dloss, d_opt], feed_dict=feed_dict)
                    feed_dict={input_hr: batch_hr, input_lr: batch_lr, training:True}
                    g_loss,gpsnr,_ = sess.run([perceptual_loss,gen_psnr,g_opt], feed_dict=feed_dict)
                    dloss_arr.append(d_loss)
                    gloss_arr.append(g_loss)
                    print('Loss ', count, '--', global_step.eval(), '--', ' | dloss --> ', d_loss, ' | gloss --> ', g_loss, ' | PSNR -->', gpsnr)

                if (count % 250) == 0:
                    print('Validation Set');
                    feed_dict={input_hr: get_val_hr(), input_lr: get_val_lr(), training:False}
                    [vdloss, vgloss, vgpsnr] = sess.run([dloss,perceptual_loss,gen_psnr], feed_dict=feed_dict)
                    print([vdloss, vgloss, vgpsnr])
                    
                    vdloss_arr.append(vdloss)
                    vgloss_arr.append(vgloss)

                    vdlossfile = open('vdloss.txt', 'w')
                    vglossfile = open('vgloss.txt', 'w')

                    vdlossfile.write(str(vdloss_arr))
                    vglossfile.write(str(vgloss_arr))

                    vdlossfile.close()
                    vglossfile.close()

                # if (count % 5000) == 0: #Testing 
                #     do_test(sess, count, input_hr, input_lr, gen_hr)

                if (count % 1000) == 0:
                    saver.save(sess, model_dir + model_name, global_step.eval())

                    dlossfile = open('dloss.txt', 'w')
                    glossfile = open('gloss.txt', 'w')
                    clossfile = open('closs.txt', 'w')

                    dlossfile.write(str(dloss_arr))
                    glossfile.write(str(gloss_arr))
                    clossfile.write(str(closs_arr))

                    dlossfile.close()
                    glossfile.close()
                    clossfile.close()

            np.random.shuffle(datn)

 