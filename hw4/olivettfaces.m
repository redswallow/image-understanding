load('olivettifaces.mat');

nPos = 400;
nNeg = 500;
% get postive patches
h = 64; w = 64;
posPatches = zeros(h, w, nPos);
for i = 1:nPos
    posPatches(:,:,i) = imresize(reshape(faces(:,i),64,64),[h w]);
end
% build postive template
posTemplate = zeros(h/8, w/8, 9);
for i = 1:nPos
    posTemplate = posTemplate + hog(posPatches(:,:,i));
end
posTemplate = posTemplate/nPos;
%V = hogdraw(posTemplate);
%imshow(V);

Itrain = im2double(rgb2gray(imread('test1.jpg')));
% get negative patches
[ItrainH, ItrainW] = size(Itrain);
negPatches = zeros(h, w, nNeg);
for i = 1:nNeg
    x = randi(ItrainH - h);
    y = randi(ItrainW - w);
    negPatches(:,:,i) = Itrain(x : x + h - 1, y : y + w - 1);
end
% build negative template
negTemplate = zeros(h/8, w/8, 9);
for i = 1:nNeg
   negTemplate = negTemplate + hog(negPatches(:,:,i));
end
negTemplate = negTemplate/nNeg;
%V = hogdraw(posTemplate);
%imshow(V);
template = posTemplate - negTemplate;
V = hogdraw(template);
figure(1);imshow(V);

%
% load a test image
%
Itest= im2double(rgb2gray(imread('facetest6.jpg')));
V = hogdraw(hog(Itest));
figure(2);imshow(V);
imwrite(V,'p4_3_5.jpg');

% find top 5 detections in Itest
ndet = 4;
tsize = 4;

[x,y,score,scale] = multiscale_detect(Itest,template,ndet);
%[x,y,score] = detect(Itest,template,ndet);

%display top ndet detections
figure(3); clf; imshow(Itest);
for i = 1:ndet
  % draw a rectangle.  use color to encode confidence of detection
  %  top scoring are green, fading to red
  hold on; 
  h = rectangle('Position',[x(i)-tsize*8*scale(i) y(i)-tsize*8*scale(i) tsize*8*2*scale(i) tsize*8*2*scale(i)],'EdgeColor',[(i/ndet) ((ndet-i)/ndet)  0],'LineWidth',3,'Curvature',[0.3 0.3]); 
  %h = rectangle('Position',[x(i)-tsize*8 y(i)-tsize*8 tsize*8*2 tsize*8*2],'EdgeColor',[(i/ndet) ((ndet-i)/ndet)  0],'LineWidth',3,'Curvature',[0.3 0.3]); 
  hold off;
end