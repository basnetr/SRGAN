function [I_HR, I_LR] = crop_image(filename)
    width_hr = 128;
    height_hr = 128;
    lr_factor = 4;

    I = imread(filename);
    imheight = size(I,1);
    imwidth = size(I,2);
    
    xmin_hr = round(imwidth/2) - round(width_hr/2);
    ymin_hr = round(imheight/2) - round(height_hr/2);
    
    
    rect_hr = [xmin_hr ymin_hr width_hr-1 height_hr-1];
    
    I_HR = imcrop(I,rect_hr);
    I_LR = I_HR(1:lr_factor:height_hr, 1:lr_factor:height_hr, :);
end
