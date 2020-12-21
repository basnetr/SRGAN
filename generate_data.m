clc
clear all
close all

mkdir('train_hr')
mkdir('train_lr')
mkdir('val_hr')
mkdir('val_lr')
mkdir('test_hr')
mkdir('test_lr')

%train data
filenames = dir('ImageNet/*.png');
for i = 1:size(filenames,1)
    [HR, LR] = crop_image(['ImageNet/' filenames(i).name]);
    imwrite(HR,['train_hr/' num2str(i) '.png']);
    imwrite(LR,['train_lr/' num2str(i) '.png']);
end

%val data
filenames = dir('Set5/*.bmp');
for i = 1:size(filenames,1)
    [HR, LR] = crop_image(['Set5/' filenames(i).name]);
    imwrite(HR,['val_hr/' num2str(i) '.png']);
    imwrite(LR,['val_lr/' num2str(i) '.png']);
end

%test data
filenames = dir('Set14/*.bmp');
for i = 1:size(filenames,1)
    [HR, LR] = crop_image(['Set14/' filenames(i).name]);
    imwrite(HR,['test_hr/' num2str(i) '.png']);
    imwrite(LR,['test_lr/' num2str(i) '.png']);
end