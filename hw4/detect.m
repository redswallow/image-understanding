function [x,y,score] = detect(I,template,ndet)
% return top ndet detections found by applying template to the given image.
%   x,y should contain the coordinates of the detections in the image
%   score should contain the scores of the detections
%
x = zeros(1, ndet); y = zeros(1, ndet); score = zeros(1, ndet);
% compute the feature map for the image
f = hog(I);
nori = size(f,3);

% cross-correlate template with feature map to get a total response
R = zeros(size(f,1),size(f,2));
for i = 1:nori
  R = R + conv2(f(:,:,i), rot90(template(:,:,i),2),'same');
  %R = R + imfilter(f(:,:,i),template(:,:,i),'symmetric');
end

% now return locations of the top ndet detections

% sort response from high to low
[val,ind] = sort(R(:),'descend');

% work down the list of responses, removing overlapping detections as we go
i = 1;
detcount = 0;

while ((detcount < ndet) && (i < length(ind)))
  % convert ind(i) back to (i,j) values to get coordinates of the block
  yblock = mod(ind(i)-1, size(f,1)) + 1;
  xblock = floor((ind(i)-1)/size(f,1)) + 1;
  
  assert(val(i)==R(yblock,xblock)); %make sure we did the indexing correctly

  % now convert yblock,xblock to pixel coordinates 
  ypixel = yblock * 8;
  xpixel = xblock * 8;

  % check if this detection overlaps any detections which we've already added to the list
  if (detcount == 0)
      overlap = 0;
  else
      dist = (y-ypixel).^2 + (x-xpixel).^2;
      overlap = sum(dist < (16*8).^2);% ... 
      %&& xpixel + size(f,1)*8 > size(I, 1) && xpixel - size(f,1)*8 < 0 ...
      %&& ypixel + size(f,2)*8 > size(I, 2) && ypixel - size(f,2)*8 < 0;
  end
  
  % if not, then add this detection location and score to the list we return
  if (~overlap)
    detcount = detcount+1;
    x(detcount) = xpixel;
    y(detcount) = ypixel;
    score(detcount) = R(yblock,xblock);
  end
  i = i + 1;
end


