clc
clear all
close all

model_num = 0;
psnr = zeros(8,1);
for i = 0:7
    sr_im = imread(['test_op_' num2str(model_num) '/' num2str(i) '.png']);
    hr_im = imread(['test_hr/' num2str(i+1) '.png']);
    err = (sr_im - hr_im).^2;
    mse = mean(err(:));
    psnr(i+1) = 10*log(255*255/mse)/log(10);
end
mean(psnr)
