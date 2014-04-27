function [patch, X, Y] = patch(I)
    [X, Y] = ginput(I,2);
    % get patch image
    patch = I(Y(1):Y(2),X(1):X(2));
end