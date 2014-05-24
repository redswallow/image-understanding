% load a training example image
Itrain = im2double(rgb2gray(imread('test2.jpg')));

%have the user click on some training examples.  
% If there is more than 1 example in the training image (e.g. faces), you could set nclicks higher here and average together
nclick = 1;
figure(1); clf;
imshow(Itrain);
[x,y] = ginput(nclick); %get nclicks from the user

%compute 8x8 block in which the user clicked
blockx = round(x/8);
blocky = round(y/8); 

tsize = 8;

%visualize image patches that the user clicked on
figure(2); clf;
for i = 1:nclick
  patch = Itrain(8*blocky(i)+(-tsize*8+1:tsize*8),8*blockx(i)+(-tsize*8+1:tsize*8));
  figure(2); subplot(3,2,i); imshow(patch);
end

% compute the hog features
f = hog(Itrain);

% compute the average template for the user clicks
template = zeros(tsize*2,tsize*2,9);
for i = 1:nclick
  template = template + f(blocky(i)+(-tsize+1:tsize),blockx(i)+(-tsize+1:tsize),:); 
end
template = template/nclick;
V = hogdraw(template);
imshow(V);
imwrite(V, 'pedtemplate1.jpg');

%
% load a test image
%
Itest= im2double(rgb2gray(imread('test1.jpg')));


% find top 5 detections in Itest
ndet = 5;

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
