*** INSTRUCTIONS TO RUN THE CODE ***

Download ImageNet (train), Set5 (val) and Set14 (test) data.

Extract the ImageNet.zip, Set5.zip and Set14.zip files with images inside (no subdirectories)

Run "generate_data.m" in MATLAB to generate 128x128 High-Resolution files and 32x32 Low-Resolution files for train, val and test.

Code is implemented in Python 3 with following dependencies:
	Tensorflow 1.8.0
	Numpy 1.14.5
	imageio 2.3.0

Then run "srgan.py" for training.

The training parameters for the codes are:
	batch_size = 8; (batch size)
	hr_size = 128; (high resolution size)
	lr_size = 32; (low resolution size)
	nchannels = 3; (number of channels - RGB)
	lr = 0.0001 (learning_rate)

-> Training saves model parameters in the folder "srgan_models" for every 1000 iterations.
-> It also generates "closs.txt", "dloss.txt", "gloss.txt" which stores the content loss, discriminator loss and perceptual loss respectively for train data and "vdloss.txt", "vgloss.txt" which stores the discriminator loss and perceptual loss respectively for validation data.
--> Validation is done every 250 iterations (can be changed).
--> The model is saved every 1000 iterations which can be changed.
--> To load a model at certain iteration, the load_model variable can be assigned the value of the number of iterations the training needs to be resumed from (given that the model at that iteration was saved before)
--> If you want to download a pretrained model (for 30,000 iterations) and test, you can download it from the link https://drive.google.com/open?id=1sZCe0QLYYicJjZnMULOvEBZJWWHriydO

[**Note: The code initially trains the Generator (also known as SRResNet for 20,000 iterations and then trains tha SRGAN network for the rest of the iterations)]

Run "srgan_test.py" for testing.
--> Assign the model number to the variable model_num and run.
--> This will generate the Super-Resolved files in the folder test_op_[model_num]

To calculate the PSNR set the model_num variable and run "testpsnr.m" 
