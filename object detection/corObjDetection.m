function [X,Y] = corObjDetection(I, patch, Xpatch, Ypatch, threshold)
    [height, width] = size(I);
    % apply gaussian smooth on the patch image
    gauFilter = fspecial('gaussian',[3 3], 0.5);
    patchSmooth = imfilter(patch, gauFilter, 'replicate');
    % compute correlation
    Tsquared = sum(sum(patchSmooth.^2));
    Isquared = conv2(I.^2,ones(size(patchSmooth)),'same');
    Icor = conv2(I, fliplr(flipud(patchSmooth)),'same');
    Icor = Icor./sqrt(Tsquared)./sqrt(Isquared);
    % zero out any pixels which are smaller than one of their 4 neighbors
    xl = [zeros(height,1) Icor(:,2:end-1) > Icor(:,1:end-2) zeros(height,1)];
    xr = [zeros(height,1) Icor(:,2:end-1) > Icor(:,3:end) zeros(height,1)];
    yl = [zeros(1, width); Icor(2:end-1,:) > Icor(1:end-2,:); zeros(1, width)];
    yr = [zeros(1, width); Icor(2:end-1,:) > Icor(3:end,:); zeros(1, width)];
    t = Icor > threshold;
    % get points above threshold local maxima
    maxima = xl & xr & yl & yr & t;
    [X, Y] = find(maxima==1);
    % plot matched points
    imshow(I), hold on;
    plotRec(Xpatch, Ypatch);
    plot(Y, X, 'rx', 'markersize', 15, 'linewidth', 5);
    %imagesc(Icor); colorbar;
end