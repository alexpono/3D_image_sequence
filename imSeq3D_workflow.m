%% Big workflow for ESRF data

% améliorer % show a wall
clear all, close all

name = getenv('COMPUTERNAME');
if strcmp(name,'DESKTOP-3ONLTD9')
    nameFolder = strcat('C:\Users\Lenovo\Jottacloud\RECHERCHE\',...
                        'Projets\02_ESRF\',...
                        'data\ESRF\pinus02-y-P2\');
    cd(nameFolder)
    nameFile = 'pinus02-y-P2_0072.tif';
elseif strcmp(name,'DARCY')
    nameFolder = strcat('E:\ponoCleaningHDD\PRO\PRO_ESRF\ESRF_data\',...
                        'pinus02-t-P2_out\');
    cd(nameFolder)
    nameExpe = 'pinus02-t-P2';
    nameFile = 'pinus02-q-P2_0000.tif';
end


folderFile = strcat(nameFolder,nameFile);
% 
% tiff_info = imfinfo('2-A^11815^52071.tif'); % return tiff structure, one element per image
% tiff_stack = imread('2-A^11815^52071.tif', 1) ; % read in first image
% %concatenate each successive tiff to tiff_stack
% for ii = 2 : size(tiff_info, 1)
%     temp_tiff = imread('2-A^11815^52071.tif', ii);
%     tiff_stack = cat(3 , tiff_stack, temp_tiff);
% end
% 
% %write a Tiff file, appending each image as a new page
% for ii = 1 : size(tiff_stack, 3)
%     imwrite(tiff_stack(:,:,ii) , 'new_stack.tif' , 'WriteMode' , 'append') ;
% end
%% SAVING DATA
cd(nameFolder)
save('pinus02-y-P2_0072.mat','walls','pits','channels')
%% LOADING DATA
cd(nameFolder)
load('pinus02-y-P2_0072.mat')

%% read a 3D stack and display plane at z = 1008
c = clock; fprintf('start loading 2D stack at %0.2dh%0.2dm\n',c(4),c(5))
% load the 3D stack

iz = 1008;

it = 00;
file00 = strcat(nameFolder,nameExpe,sprintf('_%0.4d.tif',it));
tiff_info = imfinfo(file00);
clear im2D
im2D00 = imread(file00, iz);

it = 72;
file72 = strcat(nameFolder,nameExpe,sprintf('_%0.4d.tif',it));
tiff_info = imfinfo(file72);
clear im2D
im2D72 = imread(file72, iz);

c = clock; fprintf('3D stack read at %0.2dh%0.2dm\n',c(4),c(5))

imDiff = imsubtract(im2D00,im2D72);
figure('defaultAxesFontSize',20)
imagesc(imDiff>6)
stats = regionprops(imDiff>6,'Area','Centroid','ConvexHull');

figure
imshow(im2D72)
hold on
listRegions = find([stats.Area]>25);
for ilr = listRegions
    %plot(stats(ilr).ConvexHull(:,1),stats(ilr).ConvexHull(:,2),'or-')
    clear V F
    V = [stats(ilr).ConvexHull(:,1),stats(ilr).ConvexHull(:,2)];
    F = [1:size(V,1)];
    patch('Faces',F,'Vertices',V,...
        'faceColor',[0.1 0.1 0.8],'faceAlpha',.3,'edgeColor','none')
end
%%
hf = figure('defaultAxesFontSize',20);
set(gcf,'position',[10 10 900 900])
him = imshow(im2D00);
%for it = 0 : 73
it = 73;
while(1)
    if it == 0
        it = 73;
        figure(hf)
        title('after')
    elseif it == 73
        it = 0;
        figure(hf)
        title('before')
    end
    pause(.2)
    fileIm = strcat(nameFolder,nameExpe,sprintf('_%0.4d.tif',it));
    tiff_info = imfinfo(file00);
    clear im2D
    im2D = imread(fileIm, iz);
    him.CData = im2D;
end
%% read a 3D Stack in any plane
c = clock; fprintf('start loading 2D stack at %0.2dh%0.2dm\n',c(4),c(5))
% load the 3D stack
tiff_info = imfinfo(folderFile);
clear im2D
iz = 1;
im2D = imread(folderFile, iz);
im3D = zeros(size(im2D,1),size(im2D,2),size(tiff_info,1),class(im2D));
for iz = 1 : size(tiff_info, 1)
    clear im2D
    im2D = imread(folderFile, iz);
    im3D(:,:,iz) = im2D;
end
c = clock; fprintf('3D stack read at %0.2dh%0.2dm\n',c(4),c(5))
%% Segment one image
% walls         % delimiting channel n° and n°
% channels      % filled/empty
% pits
 
initialize = 'no'; % 'yes'/'no'
% to do only once
switch initialize
    case 'yes'
walls       = struct();
channels    = struct();
pits        = struct();
channels(1).hmx = [];       % hand measurement x
channels(1).hmy = [];       % hand measurement y
channels(1).hmslice = [];   % hand measurement slice
channels(2).hmx = [];       % hand measurement x
channels(2).hmy = [];       % hand measurement y
channels(2).hmslice = [];   % hand measurement slice
end
%%
walls(1).ch = [1,2];

for ich = 1 : size(channels,2)
    channels(ich).z = min([channels(ich).hmslice]) : max([channels(ich).hmslice]);
    channels(ich).x = interp1([channels(ich).hmslice],[channels(ich).hmx],[channels(ich).z]);
    channels(ich).y = interp1([channels(ich).hmslice],[channels(ich).hmy],[channels(ich).z]);
end

for iw = 1 : size(walls,2)
    [zz,zza,zzb] = intersect(channels(walls(iw).ch(1)).z,channels(walls(iw).ch(2)).z);
    walls(iw).sChA = zza; % curviline abscisse in channel A
    walls(iw).sChB = zzb; % curviline abscisse in channel B
    walls(iw).x = ( (channels(walls(iw).ch(1)).x(zza)) + (channels(walls(iw).ch(2)).x(zzb)) ) / 2;
    walls(iw).y = ( (channels(walls(iw).ch(1)).y(zza)) + (channels(walls(iw).ch(2)).y(zzb)) ) / 2;
    walls(iw).z = zz;
end

figure('defaultAxesFontSize',20), hold on, box on
for ich = 1 : size(channels,2)
    plot3([channels(ich).hmx],[channels(ich).hmy],[channels(ich).hmslice])
    plot3([channels(ich).x],[channels(ich).y],[channels(ich).z],'o')
end
for iw = 1 : size(walls,2)
    plot3([walls(iw).x],[walls(iw).y],[walls(iw).z],'-og')
end

view(3)
xlabel('x')
ylabel('y')
zlabel('z')

%% show a wall and cut perpendicular to wall 

tic
iw = 1;

clear imWall
imWall = struct();

clear imWallPerp
imWallPerp = struct();

%[X,Y] = meshgrid(1:size(im2D,1));
for iz = 1 : length(walls(iw).z)
    
clear im2D
im2D = zeros(336,336,'uint8');
im2D = im3D(:,:,walls(1).z(iz));
%figure('defaultAxesFontSize',20), hold on, box on
%imagesc(im2D), colormap gray
%set(gca,'ydir','reverse')
iA = walls(iw).sChA(iz);
iB = walls(iw).sChB(iz);
xA = channels(walls(iw).ch(1)).x(iA);
yA = channels(walls(iw).ch(1)).y(iA);
xB = channels(walls(iw).ch(2)).x(iB);
yB = channels(walls(iw).ch(2)).y(iB);
%plot(xA,yA,'ob')
%plot(xB,yB,'or')
%axis([-20+min(xA,xB) 20+max(xA,xB) -20+min(yA,yB) 20+max(yA,yB)])

theta = 90;
R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
vecAB = [xA,yA]-[xB,yB];
uAB = [xB-xA,yB-yA]/(norm(vecAB));
vAB = R * uAB';

% figure, hold on, box on
% quiver(0,0,uAB(1),uAB(2))
% quiver(0,0,vAB(1),vAB(2))
% axis equal

clear lineVal lineValPerp 

smax = (ceil( norm(vecAB) ));
for is = 1 : smax
    % is = 9
    xs = (xA + (xB-xA)*is/smax);
    ys = (yA + (yB-yA)*is/smax);
    
    inn = 0;
    clear pxl
    for in = -2 : 2
        inn = inn + 1;
        Xq = xs+in*vAB(1);
        Yq = ys+in*vAB(2);
        pxl(inn) = PONOsubpix(im2D,Xq,Yq);
    end
    lineVal(is) = min(pxl);
end
imWall(iz).line = lineVal;
imWall(iz).lineLength = length(lineVal);

wPerp = [-10:1:10];
for ip = 1 : length(wPerp)
    xs = xA + wPerp(ip) * vAB(1);
    ys = yA + wPerp(ip) * vAB(2);
    inn = 0;
    clear pxl
    for in = -7 : 7
        inn = inn + 1;
        Xq = xs+in*uAB(1);
        Yq = ys+in*uAB(2);
        pxl(inn) = PONOsubpix(im2D,Xq,Yq);
    end
    lineValPerp(ip) = mean(pxl);
end
imWallPerp(iz).line = lineValPerp;
imWallPerp(iz).lineLength = length(lineValPerp);

end

toc

imWallVar = zeros(length(walls(iw).z) , max([imWall.lineLength]),'uint8');
for iz = 1 : length(walls(iw).z)
    imWallVar(iz,1:imWall(iz).lineLength) = imWall(iz).line;
end
figure('defaultAxesFontSize',20), hold on, box on
imagesc(imWallVar'), colormap gray

imWallPerpVar = zeros(length(walls(iw).z) , max([imWallPerp.lineLength]),'uint8');
for iz = 1 : length(walls(iw).z)
    imWallPerpVar(iz,1:imWallPerp(iz).lineLength) = imWallPerp(iz).line;
end
figure('defaultAxesFontSize',20), hold on, box on
imagesc(imWallPerpVar'), colormap gray


%% on the image perpendicular to the wall I can indicate where is the wall
imWallVar = zeros(length(walls(iw).z) , max([imWall.lineLength]),'uint8');
for iz = 1 : length(walls(iw).z)
    imWallVar(iz,1:imWall(iz).lineLength) = imWall(iz).line;
end
figure('defaultAxesFontSize',20), hold on, box on
title('a: continue, z: new wall')
imagesc(imWallVar'), colormap gray
set(gcf,'position',[99         718        1818         253])
set(gcf,'currentchar','a')         % set a dummy character
wallwall = struct();
iw = 1;
wallwall(iw).x = [];
wallwall(iw).y = [];
while get(gcf,'currentchar')=='a' || get(gcf,'currentchar')=='z'  % which gets changed when key is pressed
   [x,y] = ginput(1);
   if get(gcf,'currentchar')=='a'
       plot(x,y,'og')
       wallwall(iw).x = [wallwall(iw).x,y];
       wallwall(iw).y = [wallwall(iw).y,x];
   elseif get(gcf,'currentchar')=='z'
       iw = iw + 1;
       wallwall(iw).x = y;
       wallwall(iw).y = x;
       plot(x,y,'or')
   end
   fprintf('x: %0.0f, y: %0.0f \n',x,y)
end
fprintf('finished \n')

%% passer de x,y à [x,y,z]
iw = 1;
iz = 60;
is = 7;

iA = walls(iw).sChA(iz);
iB = walls(iw).sChB(iz);
xA = channels(walls(iw).ch(1)).x(iA);
yA = channels(walls(iw).ch(1)).y(iA);
xB = channels(walls(iw).ch(2)).x(iB);
yB = channels(walls(iw).ch(2)).y(iB);

vecAB = [xA,yA]-[xB,yB];
uAB = [xB-xA,yB-yA]/(norm(vecAB));



xs = (xA + (xB-xA)*is/smax)
ys = (yA + (yB-yA)*is/smax)
walls(iw).z(iz)


%% navigate in 3D
close all


him = figure('defaultAxesFontSize',20); hold on, box on
iz = 1000;
him2D = imshow(im3D(:,:,iz));
xAxis = [0,336];
yAxis = [0,336];
clear xCH yCH
xCH = [];
yCH = [];
for ich = 1 : length(channels)
    idx = find(iz == channels(ich).z);
    xCH(ich) = channels(ich).x(idx);
    yCH(ich) = channels(ich).y(idx);
end
figure(him), hold on
hCH = plot(xCH,yCH,'ob');
set(gcf,'currentchar','a')         % set a dummy character
set(gcf,'position',[100 100 900 800])
title(sprintf('z: %0.0f',iz))

hminiMap = figure('defaultAxesFontSize',20); hold on, box on
plot3([0,336,336,0,0,0,336,336,0,0],[0,0,336,336,0,0,0,336,336,0],2016*[0,0,0,0,0,1,1,1,1,1],'.k')
X = [0,336,336,0,0];
Y = [0,0,336,336,0];
Z = iz * [1,1,1,1,1];
hp = patch('XData',X,'YData',Y,'ZData',Z);
hp.FaceColor = 'r';
hp.EdgeColor = 'none';
hp.FaceAlpha = .5;
set(gcf,'position',[1100 100 200 800])
caz =  -36.4150;
cel =    7.2736;
view(caz,cel)
   hold on
   for ich = 1 : size(channels,2)
       plot3([channels(ich).x],[channels(ich).y],[channels(ich).z],'o')
   end
%camtarget([200,200,1250])
%camva(9.33)
%campos([-2291,-2190,3042])
while contains('azesdxcr',get(gcf,'currentchar'))  % which gets changed when key is pressed
   figure(him)
   [x,y] = ginput(1);
   if get(gcf,'currentchar')=='z'
       iz = iz -  1;
   elseif get(gcf,'currentchar')=='e'
       iz = iz +  1;
   elseif get(gcf,'currentchar')=='s'
       iz = iz - 10;
   elseif get(gcf,'currentchar')=='d'
       iz = iz + 10;
   elseif get(gcf,'currentchar')=='x'
       iz = iz - 100;
   elseif get(gcf,'currentchar')=='c'
       iz = iz + 100;
    elseif get(gcf,'currentchar')=='r' 
       xAxis(1) = x-30; xAxis(2)=x+30; yAxis(1)=y-30; yAxis(2)=y+30;
   elseif get(gcf,'currentchar')=='w' % start a new wall measurement 
       % ginput the two channels
       % got to the smallest z where they exist together 
   end
   
   clear xCH yCH
   xCH = [];
   yCH = [];
   for ich = 1 : length(channels)
       idx = find(iz == channels(ich).z);
       if idx
           xCH(ich) = channels(ich).x(idx);
           yCH(ich) = channels(ich).y(idx);
       end
   end
   
   
   figure(him), hold on
   im2D = im3D(:,:,iz);
   him2D.CData = im2D;
   axis(gca,[xAxis(1) xAxis(2) yAxis(1) yAxis(2)])
   hCH.XData = xCH;
   hCH.YData = yCH;
   %set(gcf,'position',[100 100 1000 1000])
   title(sprintf('z: %0.0f',iz))
   figure(hminiMap)
   hp.Vertices(:,3) = iz * [1,1,1,1,1];

end

%%
% img = imread('corn.tif');
% trees.tif
 img = imread('westconcordorthophoto.png');
    imshow(img); hold on
set(gcf,'currentchar','a')         % set a dummy character
while get(gcf,'currentchar')=='a' || get(gcf,'currentchar')=='z'  % which gets changed when key is pressed
   [x,y] = ginput(1);
   if get(gcf,'currentchar')=='a'
       plot(x,y,'og')
   elseif get(gcf,'currentchar')=='z'
       plot(x,y,'or')
   end
   fprintf('x: %0.0f, y: %0.0f \n',x,y)
end
fprintf('finished \n')
%% Mark pits


PONOsubpix(im2D,Xq,Yq)

%%

%% from https://fr.mathworks.com/matlabcentral/answers/753419-theory-of-bicubic-interpolation
% see also https://fr.mathworks.com/matlabcentral/answers/78116-algorithm-of-bicubic-interpolation
% or https://thilinasameera.wordpress.com/2010/12/24/digital-image-zooming-sample-codes-on-matlab/

x = [-1:1];
y = [-1:1];
[X,Y] = meshgrid(x,y);
xq = rand;
yq = rand;
Z = rand(size(X));
Zq = interp2(x,y,Z,xq,yq,'bicubic')
% Check bicubic formula
Pl = [1.5,-2.5,0,1];
Pr = [-0.5,2.5,-4,2];
cubicp = @(x) (x<=1).*polyval(Pl,x) + (x>1 & x<2).*polyval(Pr,x);
cubic = @(x) cubicp(abs(x));
Zq = 0;
[~,i0] = histc(xq,x);
[~,j0] = histc(yq,y);
for i = i0-1:i0+2
    for j = j0-1:j0+2
        k = sub2ind(size(Z),j,i);
        Zq = Zq + cubic(X(k)-xq)*cubic(Y(k)-yq)*Z(k);
    end
end
Zq


%% functions

function val = PONOsubpix(im2D,Xq,Yq)

% val = im2D(round(Yq),round(Xq));

% Z = uint8(imcrop(im2D,[floor(Yq-1),floor(Xq-1),2,2]));
% x = [-1:1];
% y = [-1:1];
% val = interp2(x,y,Z,Xq,Yq,'bicubic');

if Xq == round(Xq) && Yq == round(Yq)
    val = im2D(round(Yq),round(Xq));
else
    xA = floor(Xq);
    yA = floor(Yq);
    xB = ceil(Xq);
    yB = floor(Yq);
    xC = floor(Xq);
    yC = ceil(Yq);
    xD = ceil(Xq);
    yD = ceil(Yq);
    
    coeffA = abs(xB-Xq) * abs(yC-Yq);
    coeffB = abs(xA-Xq) * abs(yC-Yq);
    coeffC = abs(xD-Xq) * abs(yA-Yq);
    coeffD = abs(xC-Xq) * abs(yA-Yq);
    
    val = coeffA * im2D(yA,xA) + coeffB * im2D(yB,xB) + ...
        coeffC * im2D(yC,xC) + coeffD * im2D(yD,xD);
end

end


