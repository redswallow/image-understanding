function [mag,ori] = mygradient(I)
%
% compute image gradient magnitude and orientation at each pixel
%
gaussianD = fspecial('sobel');
dx = imfilter(I, gaussianD);
dy = imfilter(I, gaussianD');
mag = sqrt(dx.*dx + dy.*dy);
ori = atan2(dy, dx);