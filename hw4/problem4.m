% load a training example image
Itrain = im2double(rgb2gray(imread('test2.jpg')));
%figure(1); clf;
%imshow(Itrain);
%rect = getrect;
%imwrite(imcrop(Itrain,rect),'p41p1.jpg');

nPos = 5;
nNeg = 100;
% get postive patches
patches = cell(1,nPos);
h = zeros(1,nPos); w = zeros(1,nPos);
patches{1} = im2double((imread('p41p1.jpg')));
patches{2} = im2double((imread('p41p2.jpg')));
patches{3} = im2double((imread('p41p3.jpg')));
patches{4} = im2double((imread('p41p4.jpg')));
patches{5} = im2double((imread('p41p5.jpg')));
% compute templateH, templateW
for i = 1:nPos
    [h(i), w(i)] = size(patches{i});
end
aspectRatio = w./h;
templateH = mean(h);
templateW = templateH * mean(aspectRatio);
templateH = floor(templateH/8) * 8;
templateW = floor(templateW/8) * 8;
%templateH = 32; templateW = 32;
% resize patches
posPatches = zeros(templateH, templateW, nPos);
for i = 1:nPos
    posPatches(:,:,i) = imresize(patches{i},[templateH templateW]);
end
% get negative patches
[ItrainH, ItrainW] = size(Itrain);
negPatches = zeros(templateH, templateW, nNeg);
for i = 1:nNeg
    x = randi(ItrainH - templateH);
    y = randi(ItrainW - templateW);
    negPatches(:,:,i) = Itrain(x : x + templateH - 1, y : y + templateW - 1);
end
% build postive template
posTemplate = zeros(floor(templateH/8),floor(templateW/8),9);
for i = 1:nPos
   posTemplate = posTemplate + hog(posPatches(:,:,i));
end
posTemplate = posTemplate/nPos;
%V = hogdraw(posTemplate);
%imshow(V);
% build negative template
negTemplate = zeros(floor(templateH/8),floor(templateW/8),9);
for i = 1:nNeg
   negTemplate = negTemplate + hog(negPatches(:,:,i));
end
negTemplate = negTemplate/nNeg;
%V = hogdraw(posTemplate);
%imshow(V);
template = posTemplate - negTemplate;
V = hogdraw(template);
imshow(V);
imwrite(V, 'pedtemplate3.jpg');
%
% load a test image
%
Itest= im2double(rgb2gray(imread('test3.jpg')));
V = hogdraw(hog(Itest));
imshow(V);

% find top 5 detections in Itest
ndet = 5;
tsize = 8;
%[x,y,score,scale] = multiscale_detect(Itest,template,ndet);
[x,y,score] = detect(Itest,template,ndet);

%display top ndet detections
figure(3); clf; imshow(Itest);
for i = 1:ndet
  % draw a rectangle.  use color to encode confidence of detection
  %  top scoring are green, fading to red
  hold on; 
  %h = rectangle('Position',[x(i)-tsize*8*scale(i) y(i)-tsize*8*scale(i) tsize*8*2*scale(i) tsize*8*2*scale(i)],'EdgeColor',[(i/ndet) ((ndet-i)/ndet)  0],'LineWidth',3,'Curvature',[0.3 0.3]); 
  h = rectangle('Position',[x(i)-tsize*8 y(i)-tsize*8 tsize*8*2 tsize*8*2],'EdgeColor',[(i/ndet) ((ndet-i)/ndet)  0],'LineWidth',3,'Curvature',[0.3 0.3]); 
  hold off;
end
