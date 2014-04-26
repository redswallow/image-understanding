function [gauss_x, gauss_y] = gauss2d(sigma)
    dx = floor(3.0*sigma);
    dy = floor(3.0*sigma);
    [X, Y] = meshgrid(-dx:dx,-dy:dy);
    k = 1/sqrt(2*pi*sigma^2);
    gx = k*exp(-0.5*(X.^2/sigma^2));
    gy = k*exp(-0.5*(Y.^2/sigma^2));
    g = k*exp(-0.5*((X.^2+Y.^2)/sigma^2));
    gauss_x = (-X./(sigma^2)).*g.*gy;
    gauss_y = (-Y./(sigma^2)).*g.*gx;
end