function [gradMag, gradDir] = edgeDetect(sigma)
    % gaussian smoothing
    H = fspecial('gaussian',[3 3], sigma);
    I = imfilter(A, H, 'replicate');
    % get the derivative filter
    [gauss_x, gauss_y] =gauss2d(sigma);
    horgradI = abs(conv2(I,gauss_x,'same'));
    vergradI = abs(conv2(I,gauss_y,'same'));
    % compute magnitude
    gradMag = sqrt(horgradI.^2 + vergradI.^2);
    % compute orientation
    gradDir = atan2(vergradI, horgradI);
end