%% Big workflow for ESRF data

% améliorer % show a wall
clear all, close all

name = getenv('COMPUTERNAME');
if strcmp(name,'DESKTOP-3ONLTD9')
    nameFolder = strcat('C:\Users\Lenovo\Jottacloud\RECHERCHE\Projets\',...
                        '02_ESRF\data\ESRF_full_sequences',...
                        '\pinus02-y-P2\');
    cd(nameFolder)
    nameExpe = 'pinus02-y-P2';
    nameFile = 'pinus02-y-P2_0000.tif';
elseif strcmp(name,'DARCY')
    nameFolder = strcat('E:\ponoCleaningHDD\PRO\PRO_ESRF\ESRF_data\',...
                        'pinus02-h-P2_out\');
    cd(nameFolder)
    nameExpe = 'pinus02-h-P2';
    nameFile = 'pinus02-h-P2_0000.tif';
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

%% im subtract and regions props of 2D images
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
himTh = figure('defaultAxesFontSize',20);
imagesc(imDiff>6), colormap gray
stats = regionprops(imDiff>6,'Area','Centroid','ConvexHull','Solidity');

listRegions = find([stats.Area]>25);
clear ikill
ikill = [];
for iilr = 1 : length(listRegions)
    ilr = listRegions(iilr);
    if [stats(ilr).Solidity]<.75
        ikill = [ikill,iilr];
    end
end
listRegions(ikill) = [];

figure
imshow(im2D72)
hold on
for ilr = listRegions
    %plot(stats(ilr).ConvexHull(:,1),stats(ilr).ConvexHull(:,2),'or-')
    clear V F
    V = [stats(ilr).ConvexHull(:,1),stats(ilr).ConvexHull(:,2)];
    F = [1:size(V,1)];
    patch('Faces',F,'Vertices',V,...
        'faceColor',[0.1 0.1 0.8],'faceAlpha',.3,'edgeColor','none')
end

figure(himTh)
hold on
for ilr = listRegions
     %plot(stats(ilr).ConvexHull(:,1),stats(ilr).ConvexHull(:,2),'or-')
    clear V F
    V = [stats(ilr).ConvexHull(:,1),stats(ilr).ConvexHull(:,2)];
    F = [1:size(V,1)];
    patch('Faces',F,'Vertices',V,...
        'faceColor',[0.1 0.1 0.8],'faceAlpha',.3,'edgeColor','none')
    fprintf('id: %0.0f, solidity: %4.4d \n',...
            stats(ilr).Centroid(1,1), stats(ilr).Solidity)
end
%% show before and after
hf = figure('defaultAxesFontSize',20);
set(gcf,'position',[10 10 900 900])
him = imshow(im2D00);
%for it = 0 : 73
it = 72;
while(1)
    if it == 0
        it = 72;
        figure(hf)
        title('after')
    elseif it == 72
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
%%
%% (yz) plane im subtract and regions props of 2D images 
c = clock; fprintf('start loading 2D stack at %0.2dh%0.2dm\n',c(4),c(5))
% load the 3D stack

ix = 70;

it = 00;
clear im2D00
im2D00 = zeros(336,2016,'uint8');
im2D00(:,:) = im3D00(ix, :,:);
figure,
imagesc(im2D00)

it = 72;
clear im2D72
im2D72 = zeros(336,2016,'uint8');
im2D72(:,:) = im3D72(ix, :,:);

c = clock; fprintf('3D stack read at %0.2dh%0.2dm\n',c(4),c(5))

imDiff = imsubtract(im2D00,im2D72);
himTh = figure('defaultAxesFontSize',20);
imagesc(imDiff>6), colormap gray
stats = regionprops(imDiff>6,'Area','Centroid','ConvexHull','Solidity');

listRegions = find([stats.Area]>25);
clear ikill
ikill = [];
for iilr = 1 : length(listRegions)
    ilr = listRegions(iilr);
    if [stats(ilr).Solidity]<.75
        ikill = [ikill,iilr];
    end
end
listRegions(ikill) = [];

figure
imshow(im2D72)
hold on
for ilr = listRegions
    %plot(stats(ilr).ConvexHull(:,1),stats(ilr).ConvexHull(:,2),'or-')
    clear V F
    V = [stats(ilr).ConvexHull(:,1),stats(ilr).ConvexHull(:,2)];
    F = [1:size(V,1)];
    patch('Faces',F,'Vertices',V,...
        'faceColor',[0.1 0.1 0.8],'faceAlpha',.3,'edgeColor','none')
end

figure(himTh)
hold on
for ilr = listRegions
     %plot(stats(ilr).ConvexHull(:,1),stats(ilr).ConvexHull(:,2),'or-')
    clear V F
    V = [stats(ilr).ConvexHull(:,1),stats(ilr).ConvexHull(:,2)];
    F = [1:size(V,1)];
    patch('Faces',F,'Vertices',V,...
        'faceColor',[0.1 0.1 0.8],'faceAlpha',.3,'edgeColor','none')
    fprintf('id: %0.0f, solidity: %4.4d \n',...
            stats(ilr).Centroid(1,1), stats(ilr).Solidity)
end
%% show before and after
hf = figure('defaultAxesFontSize',20);
set(gcf,'position',[10 10 900 900])
him = imshow(im2D00);
%for it = 0 : 73
it = 72;
while(1)
    if it == 0
        it = 72;
        figure(hf)
        title('after')
    im2D = im2D00;
    elseif it == 72
        it = 0;
        figure(hf)
        title('before')
    im2D = im2D72;
    end
    pause(.2)
    him.CData = im2D;
end

%% testing a blob function
%  https://fr.mathworks.com/matlabcentral/answers/176553-blob-detection-in-matlab

im2D = ~(im3D00(:,:,1008)>115);
BBOX_OUT = [];
NUM_BLOBS = [];
LABEL = [];
%%connected component analisys
hblob = vision.BlobAnalysis;
hblob.CentroidOutputPort = false;
hblob.MaximumCount = 3500;
hblob.Connectivity = 4;
hblob.MaximumBlobArea = 6500;
hblob.MinimumBlobArea = 20;
hblob.LabelMatrixOutputPort = true;
hblob.OrientationOutputPort = true;
hblob.MajorAxisLengthOutputPort = true;
hblob.MinorAxisLengthOutputPort = true;
hblob.EccentricityOutputPort = true;
hblob.ExtentOutputPort = true;
hblob.BoundingBoxOutputPort = true;
[AREA,BBOX,MAJOR,MINOR,ORIENT,ECCEN,EXTENT,LABEL] = step(hblob,im2D);
imshow(LABEL*2^16)
numberOfBlobs = length(AREA);

set(gcf,'position',[947         155        1012         832])
figure
imshow(im3D00(:,:,1008))
set(gcf,'position',[27   197   917   769])
%%      testing another web function
% https://stackoverflow.com/questions/42626416/how-to-properly-tesselate-a-image-of-cells-using-matlab/42630616
iz = 1008;
clear x y xv yv xs ys
im = 255-im3D00(:,:,iz);
figure,
imshow(im)

imEAN = 255-uint8(mean(im3D00(:,:,iz+[-10:+10]),3));
figure
imshow(imEAN)
%%

%im = imEAN;
im = double(im3D72(:,:,iz)) - double(im3D00(:,:,iz));
min(im(:))
max(im(:))
%%
figure,imagesc(im);axis image;
%%
% blur image
sigma=2;
kernel = fspecial('gaussian',4*sigma+1,sigma);
im2=imfilter(im,kernel,'symmetric');

figure,imagesc(im2);axis image; colormap gray

%% watershed
L = watershed(max(im2(:))-im2);
L = watershed(im2 > -5);
[x,y]=find(L==0);

%drw boundaries
figure,imagesc(im3D00(:,:,iz)),axis image, colormap gray
hold on, plot(y,x,'r.')

% analyse each blob
tmp=zeros(size(im));

for i=1:max(L(:))
  ind=find(L==i);
  mask=L==i;
  [thr,metric] =multithresh(im(ind),1);
  if metric>0.7
    tmp(ind)=im(ind)>thr;
  end
end

% noise removal
tmp=imopen(tmp,strel('disk',1));
figure,imagesc(tmp),axis image


clear stats
stats = regionprops(L>0,'centroid');
figure
imagesc(im3D00(:,:,iz)), colormap gray
%hold on, plot(y,x,'r.')
hold on
for is = 1 : length(stats)
    xv(is) = stats(is).Centroid(:,1);
    yv(is) = stats(is).Centroid(:,2);
    plot(xv,yv,'+r')
end

voronoi(xv,yv)
[vx,vy] = voronoi(xv,yv);

%% try to find channels by thresholding images

im2D = imresize(imgaussfilt(im3D00(:,:,1008) , 1),2);
figure
imagesc(im2D)

[centers,radii] = imfindcircles(im2D,[7 50],'ObjectPolarity','dark');
hold on
viscircles(centers, radii,'EdgeColor','b');
%% preload images - short time scale ( 1 sec )

ci = clock; fprintf('start to read 3D stack at %0.2dh%0.2dm\n',ci(4),ci(5))

it = 00;
file00 = strcat(nameFolder,nameExpe,sprintf('_%0.4d.tif',it));
tiff_info = imfinfo(file00);
clear im2D
iz = 1;
im2D = imread(file00, iz);
im3D00 = zeros(size(im2D,1),size(im2D,2),size(tiff_info,1),class(im2D));
for iz = 1 : size(tiff_info, 1)
    clear im2D
    im2D = imread(file00, iz);
    im3D00(:,:,iz) = im2D;
end

it = 72;
file72 = strcat(nameFolder,nameExpe,sprintf('_%0.4d.tif',it));
tiff_info = imfinfo(file72);
clear im2D
iz = 1;
im2D = imread(file72, iz);
im3D72 = zeros(size(im2D,1),size(im2D,2),size(tiff_info,1),class(im2D));
for iz = 1 : size(tiff_info, 1)
    clear im2D
    im2D = imread(file72, iz);
    im3D72(:,:,iz) = im2D;
end

cf = clock; fprintf('3D stack read at %0.2dh%0.2dm in %0.0f s \n',cf(4),cf(5),etime(cf,ci))


%% SLICING in (xy) planes - im subtract and regions props of 2D images

ci = clock; fprintf('3D stack read at %0.2dh%0.2dm\n',ci(4),ci(5))
clear stats toc_Readimages toc_imdiff toc_regionprops
toc_Readimages  = 0;
toc_imdiff      = 0;
toc_regionprops = 0;
for iz = 0001 : 2016
    fprintf('iz: %0.0f\n',iz)
    clear imDiff im2D00 im2D72
    
    imDiff = zeros(336,336,'uint8');
    im2D00 = zeros(336,336,'uint8');
    im2D72 = zeros(336,336,'uint8');
    
%     it = 00;
%     file00 = strcat(nameFolder,nameExpe,sprintf('_%0.4d.tif',it));
%     tiff_info = imfinfo(file00);
%     im2D00 = imread(file00, iz);
%     
%     it = 72;
%     file72 = strcat(nameFolder,nameExpe,sprintf('_%0.4d.tif',it));
%     tiff_info = imfinfo(file72);
%     im2D72 = imread(file72, iz);

    tic
    if iz >1 && iz <2016
    im2D00 = mean(im3D00(:,:,max(iz-5,1):min(iz+5,2016)),3);
    im2D72 = mean(im3D72(:,:,max(iz-5,1):min(iz+5,2016)),3);
    else
    im2D00 = im3D00(:,:,iz);
    im2D72 = im3D72(:,:,iz);
    end
        
        toc_Readimages = toc_Readimages + toc;
    
    tic
    imDiff = imsubtract(im2D00,im2D72);
    toc_imdiff = toc_imdiff + toc;

    tic
    BW = bwareaopen(imDiff>6,25,8);
    stats2D = regionprops(BW,'Area','Centroid','ConvexHull','Solidity');
    listRegions = find([stats2D.Solidity]>.75);
    stats(iz).stats = stats2D(listRegions);
    toc_regionprops = toc_regionprops + toc;
end

c = clock; fprintf('3D stack read at %0.2dh%0.2dm in %0.0f s \n',c(4),c(5),etime(c,ci))
fprintf(strcat('time 2 Readimages : %0.0f s, \n',... 
               'time 2 Readimages : %0.0f s, \n',...
               'time 2 Readimages : %0.0f s \n'),...
            toc_Readimages,toc_imdiff,toc_regionprops)
%% 3D rendering 
figure('defaultAxesFontSize',20)
hold on, box on
for iz =  1 : 1 : 2016
    clear statsiz
    statsiz = stats(iz).stats;
    listRegions = find([statsiz.Area]>25);
    for iilr = 1 : length(listRegions)
        ilr = listRegions(iilr);
%         clear V F
%         V = [statsiz(ilr).ConvexHull(:,1),...
%              statsiz(ilr).ConvexHull(:,2),...
%              iz*ones(size(statsiz(ilr).ConvexHull(:,2)))];
%         F = [1:size(V,1)];
%         patch('Faces',F,'Vertices',V,...
%             'faceColor',[0.1 0.1 0.8],'faceAlpha',.3,'edgeColor','none')
        
        clear V F
        V = [[statsiz(ilr).ConvexHull(:,1);statsiz(ilr).ConvexHull(:,1)],...
             [statsiz(ilr).ConvexHull(:,2);statsiz(ilr).ConvexHull(:,2)],...
             [(iz-.5)*ones(size(statsiz(ilr).ConvexHull(:,2)));(iz+.5)*ones(size(statsiz(ilr).ConvexHull(:,2)))]];
        clear nPts, nPts = size(V,1)/2;
        for iff = 1 : nPts-1
            F(iff,1:4) = [iff,iff+1,nPts+iff+1,nPts+iff+0];
        end
        patch('Faces',F,'Vertices',V,...
            'faceColor',[0.1 0.1 0.8],'faceAlpha',.3,'edgeColor','none')
        
    end
end
xlabel('x')
ylabel('y')
zlabel('z')
view(109,9.7)
axis([0 336 0 336 0 2016])
%axis([100 250 50 200 1450 1550])
%% show the results as an image sequence

%% straight away 3D
tic
imDiff3D = imsubtract(im3D00,im3D72);
toc
%%
figure
histogram(imDiff3D(:))
%% reginprops3 -> 8 minutes , with 'all' option: 76 minutes 
tic
%stats = regionprops3(imDiff3D>20,'all');
stats = regionprops3(imDiff3D>20,'BoundingBox','Centroid','ConvexHull','Solidity','Volume','VoxelList');
toc
%% histogram of the found volumes
figure,
histogram([stats.Volume])
%% keep the largest volumes only
listVol = find([stats.Volume]>20);

%[a,b] = maxk([stats.Volume],500);
b = listVol;
figure
hold on
box on
view(3)
colors = jet(size(stats,1));
for ib = 1 : length(b)
    clear CH_XYZ CH_X CH_Y CH_Z vDT C v 
    CH_XYZ = [stats.ConvexHull{b(ib),1}  ];
    CH_X = CH_XYZ(:,1);
    CH_Y = CH_XYZ(:,2);
    CH_Z = CH_XYZ(:,3);
    %plot3(CH_X,CH_Y,CH_Z,'ob')
    
    vDT = delaunayTriangulation(CH_X,CH_Y,CH_Z);
    [C,v] = convexHull(vDT);
    trisurf(C,vDT.Points(:,1),vDT.Points(:,2),vDT.Points(:,3), ...
        'FaceColor',colors(b(ib),:),'faceAlpha',.25);
end

%% choose a volume and display its projections on (xy) (yz) and (xz) planes 
[a,b] = maxk([stats.Volume],20);
ib = 1;
clear CH_XYZ CH_X CH_Y CH_Z vDT C v
CH_XYZ = [stats.ConvexHull{b(ib),1}  ];
CH_X = CH_XYZ(:,1);
CH_Y = CH_XYZ(:,2);
CH_Z = CH_XYZ(:,3);
%plot3(CH_X,CH_Y,CH_Z,'ob')

vDT = delaunayTriangulation(CH_X,CH_Y,CH_Z);
[C,v] = convexHull(vDT);
trisurf(C,vDT.Points(:,1),vDT.Points(:,2),vDT.Points(:,3), ...
    'FaceColor',colors(b(ib),:),'faceAlpha',.25);
box on, xlabel('x'), ylabel('y'), zlabel('z')

% list pixels in (xy) plane:
allpix_XYZ = [stats.VoxelList{b(ib),1}];
listXY = zeros(size(allpix_XYZ,1),2,'double');
listXY(:,1) = allpix_XYZ(:,1);
listXY(:,2) = allpix_XYZ(:,2);
figure
plot(listXY(:,1),listXY(:,2),'ob')
xlabel('x')
ylabel('y')

% list pixels in (xz) plane:
allpix_XYZ = [stats.VoxelList{b(ib),1}];
listXZ = zeros(size(allpix_XYZ,1),2,'double');
listXZ(:,1) = allpix_XYZ(:,1);
listXZ(:,2) = allpix_XYZ(:,3);
figure
plot(listXZ(:,1),listXZ(:,2),'ob')
xlabel('x')
ylabel('z')

% list pixels in (yz) plane:
allpix_XYZ = [stats.VoxelList{b(ib),1}];
listYZ = zeros(size(allpix_XYZ,1),2,'double');
listYZ(:,1) = allpix_XYZ(:,2);
listYZ(:,2) = allpix_XYZ(:,3);
figure
plot(listYZ(:,1),listYZ(:,2),'ob')
xlabel('y')
ylabel('z')

%%
figure, box on
plot3(allpix_XYZ(:,1),allpix_XYZ(:,2),allpix_XYZ(:,3),'ks')
%xlim([170 200])
%ylim([90 130])
%zlim([1480 1520])



%% sub volumes with labels using BONES
selVol = struct(); % selected volumes
bones = struct();

bones(1).x = [181 177 173]; 
bones(1).y = [139 105 74];
bones(1).z = [1501 1498 1502];

bones(2).x = [189 189]; 
bones(2).y = [98 98];
bones(2).z = [1490 1505];

bones(3).x = [189 189]; 
bones(3).y = [113 113];
bones(3).z = [1490 1505];

figure('defaultAxesFontSize',20,'position',[500 100 1200 800]), hold on, box on
plot3(allpix_XYZ(:,1),allpix_XYZ(:,2),allpix_XYZ(:,3),'rs')
xlim([160 200])
ylim([060 150])
zlim([1490 1505])
for ib = 1 : size(bones,2)
    selVol(ib).x = [];
    selVol(ib).y = [];
    selVol(ib).z = [];
    
    clear xB yB zB  sB xBs yBs zBs sB
    xB = bones(ib).x;
    yB = bones(ib).y;
    zB = bones(ib).z;
    sB = sqrt( (xB-xB(1)).^2 + (yB-yB(1)).^2 + (zB-zB(1)).^2 );
    
    bones(ib).xBs = interp1(sB,xB,[0:1:round(sB(end))],'spline');
    bones(ib).yBs = interp1(sB,yB,[0:1:round(sB(end))],'spline');
    bones(ib).zBs = interp1(sB,zB,[0:1:round(sB(end))],'spline');
    
    
    plot3(xB,yB,zB,'-g','lineWidth',12)
    plot3(xB,yB,zB,'ob','markerSize',20)
end
view(3)
xlabel('x')
ylabel('y')
zlabel('z')


%% determine who belongs to which bone
tic
listBoned = [];
ilb = 0;
list2Bone = [1 : length(allpix_XYZ(:,1))];
for ip = 1 : length(allpix_XYZ(:,1))
    clear xp yp zp
    xp = allpix_XYZ(ip,1);
    yp = allpix_XYZ(ip,2);
    zp = allpix_XYZ(ip,3);
    clear d
    for ib = 1 : size(bones,2)
        clear  xBs yBs zBs
        xBs   = bones(ib).xBs;
        yBs   = bones(ib).yBs;
        zBs   = bones(ib).zBs;
        d(ib) = min(sqrt( (xp-xBs).^2 + (yp-yBs).^2 + (zp-zBs).^2 ));
    end
    [a,b] = min(d);
    if a<20
        ilb = ilb + 1;
        listBoned(ilb,1) = ip;
        listBoned(ilb,2) = b;
        list2Bone(list2Bone==ip) = [];
        
        selVol(b).x = [selVol(b).x,xp];
        selVol(b).y = [selVol(b).y,yp];
        selVol(b).z = [selVol(b).z,zp];
    end
end
toc
%% now we have the volumes associated to the bones, we try to propagate to
% the other allpix_XYZ denoted by list2Bone
% to make it super fast, we extend the bones using selVol and bones
% structures

boneStep = 5;

figure('defaultAxesFontSize',20,'position',[500 100 1200 800]), hold on, box on
% test with bone 2:
ib = 2;
clear  xBs yBs zBs
xBs   = bones(ib).xBs;
yBs   = bones(ib).yBs;
zBs   = bones(ib).zBs;
clear xb yb zb D
xb = selVol(ib).x;
yb = selVol(ib).y;
zb = selVol(ib).z;
% refine bones
%for ib = 1 : length(xb)
    % find dir or the bone:
    dirB(1) = xBs(end)-xBs(length(xBs)-boneStep);
    dirB(2) = yBs(end)-yBs(length(yBs)-boneStep);
    dirB(3) = zBs(end)-zBs(length(zBs)-boneStep);
%end
w = null(dirB); % Find two orthonormal vectors which are orthogonal to v
[P,Q] = meshgrid(-20:20); % Provide a gridwork (you choose the size)
X = dirB(1)+xBs(end)+w(1,1)*P+w(1,2)*Q; % Compute the corresponding cartesian coordinates
Y = dirB(2)+yBs(end)+w(2,1)*P+w(2,2)*Q; %   using the two vectors in w
Z = dirB(3)+zBs(end)+w(3,1)*P+w(3,2)*Q;
surf(X,Y,Z,'edgeColor','none')
plot3(xb,yb,zb,'o','color',.2*[1 1 1])
plot3(xBs,yBs,zBs,'o--b','lineWidth',10)
%plot3()
view(3)
%% https://fr.mathworks.com/matlabcentral/answers/291485-how-can-i-plot-a-3d-plane-knowing-its-center-point-coordinates-and-its-normal
figure('defaultAxesFontSize',20,'position',[500 100 1200 800]), hold on, box on
v = [1, 2, 3];
x1 = 10; y1 = -5; z1 = 1000;
w = null(v); % Find two orthonormal vectors which are orthogonal to v
[P,Q] = meshgrid(-20:20); % Provide a gridwork (you choose the size)
X = x1+w(1,1)*P+w(1,2)*Q; % Compute the corresponding cartesian coordinates
Y = y1+w(2,1)*P+w(2,2)*Q; %   using the two vectors in w
Z = z1+w(3,1)*P+w(3,2)*Q;
surf(X,Y,Z,'edgeColor','none')
plot3(x1,y1,z1,'or')
plot3(x1+20*[0 v(1)],y1+20*[0 v(2)],z1+20*[0 v(3)],'-or')
view(3)
axis equal
%%
figure('defaultAxesFontSize',20,'position',[500 100 1200 800]), hold on, box on
%plot3(allpix_XYZ(:,1),allpix_XYZ(:,2),allpix_XYZ(:,3),'rs')

%%



% We redefine the bones:
clear  xBs yBs zBs

for ib = 1 : size(bones,2)
    ib
    clear xb yb zb D
    xb = selVol(ib).x;
    yb = selVol(ib).y;
    zb = selVol(ib).z;
    D  = sqrt((xb-xb').^2 + (zb-zb').^2 + (yb-yb').^2);
    [a,b] = max(D(:));
    [ia,ib] = ind2sub(size(D),b);
    plot3([xb(ia),xb(ib)],[yb(ia),yb(ib)],[zb(ia),zb(ib)],'-bo')
    
 
end
view(3)
%% we propagate from each bones using the new refine bone / curviligned
% absciss
ib = 2;
clear xb yb zb d
xb = selVol(ib).x; xAP = allpix_XYZ(list2Bone,1);
yb = selVol(ib).y; yAP = allpix_XYZ(list2Bone,2);
zb = selVol(ib).z; zAP = allpix_XYZ(list2Bone,3);
tic
d = sqrt((xb-xAP).^2+(yb-yAP).^2+(zb-zAP).^2);
toc
%% 3D figure
colVol = parula(5);
figure('defaultAxesFontSize',20,'position',[500 100 1200 800]), hold on, box on
%plot3(allpix_XYZ(:,1),allpix_XYZ(:,2),allpix_XYZ(:,3),'rs')
view(3)
for ib = 1 : size(bones,2)
    plot3( xB, yB, zB,'ob')
    plot3(bones(ib).xBs, bones(ib).yBs ,bones(ib).zBs, 'o-k')
end
% xlim([160 200])
% ylim([060 150])
% zlim([1490 1505])

for isel = 1 : 3
    plot3(selVol(isel).x,selVol(isel).y,selVol(isel).z,...
          'o','markerFaceColor',colVol(isel+1,:),...
          'markerEdgeColor','none')
end
view(3)

plot3(allpix_XYZ(list2Bone,1),allpix_XYZ(list2Bone,2),allpix_XYZ(list2Bone,3),...
    'o','markerFaceColor',.3*[1 1 1])

%% refine by hand
figure('defaultAxesFontSize',20,'position',[500 100 1200 800])
hold on, box on
for iz =  1493 : 1515
    if exist('h')
        if isvalid(h)
            delete(h)
        end
    end
    if exist('hgi','var')
            delete(hgi)
    end
    
    % select all pixels at iz
    listCurZ = find(allpix_XYZ(:,3) == iz);
    for isel = 1 : 3
        x2D = selVol(isel).x;
        y2D = selVol(isel).y;
        z2D = selVol(isel).z;
        listCurZ = find(z2D == iz);
         h = plot(x2D(listCurZ),y2D(listCurZ),...
              'o','markerFaceColor',colVol(isel+1,:),...
              'markerEdgeColor','none','markerSize',10);
    end
    xlim([170 200])
    ylim([80 140])
    axis  equal
   pause(.5)
%     xlim([min(allpix_XYZ(listCurZ,1)) max(allpix_XYZ(listCurZ,1))])
%     ylim([min(allpix_XYZ(listCurZ,2)) max(allpix_XYZ(listCurZ,2))])
%     axis equal
% 
%     igi = 0 ;
%     while(1)
%         [x,y] = ginput(1);
%         if x<min(allpix_XYZ(listCurZ,1))
%             break
%         end
%         % find the corresponding point.
%         clear d
%         d = sqrt((x-allpix_XYZ(listCurZ,1)).^2 + (y-allpix_XYZ(listCurZ,2)).^2);
%         [~,ip] = min(d);
%         ip = listCurZ(ip);
%         listPCut = [listPCut,ip];
%         igi = igi + 1;
%         hgi(igi) = plot(allpix_XYZ(ip,1),allpix_XYZ(ip,2),'+r');
%     end
end
%%
tic
subx = [1:336];
suby = [1:336];
subz = [1:2016];
[nX,nY,nZ] = size(imDiff3D(subx,suby,subz));
[X,Y,Z] = meshgrid(1:nX,1:nY,1:nZ); 
[f v] = isosurface(X,Y,Z,imDiff3D(subx,suby,subz),20);
%% sort f and v to build separate volumes
volSTR = struct();
f2 = f;

% look for the next no NaN line
find(isnan(f2(:,1)))
%il = 
% initialize a group
iV = 1;
listV = [];
volSTR(iV).rows  = 1;
volSTR(iV).listV = [];
listV = [f2(1,:)];
f2(1,:) = NaN;
%%
while(1)
clear idx 
idx = find(ismember(f2,listV));
if isempty(idx)
   break
end
[rowTMP,~] = ind2sub(size(f2),idx);
row = unique(rowTMP);
volSTR(iV).rows  = [volSTR(iV).rows,row'];
if length(row) == 1
    listV = unique([listV , unique(f2(row,:))])
else
    listV = unique([listV , unique(f2(row,:))'])
end
f2(row,:) = NaN;
end
volSTR(iV).listV = listV;
%%
tic
figure
patch('Faces',f,'Vertices',v,'edgeColor','none','faceAlpha',0.5)
xlabel('x')
ylabel('y')
zlabel('z')
xlim([170 195]), ylim([50 140]), zlim([1490 1510])
box on
toc
view(3)
%%
xlabel('x')
ylabel('y')
zlabel('z')
xlim([50 150]), ylim([50 150]), zlim([1147 1187])
%%
tiff_stack = rand(10,10,10);
[nX,nY,nZ] = size(tiff_stack);
[X,Y,Z] = meshgrid(1:nX,1:nY,1:nZ); 
[f v] = isosurface(X,Y,Z,tiff_stack,0.5);
patch('Faces',f,'Vertices',v,'edgeColor','none','faceAlpha',0.5)
view(3)
%%
%% SLICING in (xz) planes - im subtract and regions props of 2D images

hProgress = figure;
im3Dxz = zeros(336,336,2016,'uint8');

ci = clock; fprintf('3D stack read at %0.2dh%0.2dm\n',ci(4),ci(5))
clear stats toc_Readimages toc_imdiff toc_regionprops
toc_Readimages  = 0;
toc_imdiff      = 0;
toc_regionprops = 0;
for ix = 0001 : 336
    fprintf('ix: %0.0f\n',ix)
    clear imDiff im2D00 im2D72
    
    imDiff = zeros(336,2016,'uint8');
    im2D00 = zeros(336,2016,'uint8');
    im2D72 = zeros(336,2016,'uint8');


    tic
    if ix >1 && ix <336
    im2D00(:,:) = mean(im3D00(max(ix-5,1):min(ix+5,336),:,:),1);
    im2D72(:,:) = mean(im3D72(max(ix-5,1):min(ix+5,336),:,:),1);
    else
    im2D00(:,:) = im3D00(ix,:,:);
    im2D72(:,:) = im3D72(ix,:,:);
    end
        
    toc_Readimages = toc_Readimages + toc;
    
    tic
    imDiff = imsubtract(im2D00,im2D72);
    toc_imdiff = toc_imdiff + toc;

    tic
    BW = bwareaopen(imDiff>6,25,8);
    im3Dxz(ix,:,:) = BW;
    stats2D = regionprops(BW,'Area','Centroid','ConvexHull','Solidity');
    listRegions = find([stats2D.Solidity]>.75);
    stats(ix).stats = stats2D(listRegions);
    toc_regionprops = toc_regionprops + toc;
    
    figure(hProgress), hold on
    for ilr = listRegions
    clear V F
    V = [stats2D(ilr).ConvexHull(:,1),stats2D(ilr).ConvexHull(:,2)];
    F = [1:size(V,1)];
    patch('Faces',F,'Vertices',V,...
        'faceColor',[0.1 0.1 0.8],'faceAlpha',.3,'edgeColor','none')
    end

end

c = clock; fprintf('3D stack read at %0.2dh%0.2dm in %0.0f s \n',c(4),c(5),etime(c,ci))
fprintf(strcat('time 2 Readimages : %0.0f s, \n',... 
               'time 2 Readimages : %0.0f s, \n',...
               'time 2 Readimages : %0.0f s \n'),...
            toc_Readimages,toc_imdiff,toc_regionprops)

%% 3D rendering 
figure('defaultAxesFontSize',20)
hold on, box on
for ix =  1 : 1 : 336
    clear statsix
    statsix = stats(ix).stats;
    if isfield(statsix,'Area')
        ix
    listRegions = find([statsix.Area]>5);
    for iilr = 1 : length(listRegions)
        ilr = listRegions(iilr);
%         clear V F
%         V = [statsiz(ilr).ConvexHull(:,1),...
%              statsiz(ilr).ConvexHull(:,2),...
%              iz*ones(size(statsiz(ilr).ConvexHull(:,2)))];
%         F = [1:size(V,1)];
%         patch('Faces',F,'Vertices',V,...
%             'faceColor',[0.1 0.1 0.8],'faceAlpha',.3,'edgeColor','none')
        
        clear V F
        V = [[statsix(ilr).ConvexHull(:,1);statsix(ilr).ConvexHull(:,1)],...
             [statsix(ilr).ConvexHull(:,2);statsix(ilr).ConvexHull(:,2)],...
             [(ix-.5)*ones(size(statsix(ilr).ConvexHull(:,2)));(ix+.5)*ones(size(statsix(ilr).ConvexHull(:,2)))]];
        clear nPts, nPts = size(V,1)/2;
        for iff = 1 : nPts-1
            F(iff,1:4) = [iff,iff+1,nPts+iff+1,nPts+iff+0];
        end
        patch('Faces',F,'Vertices',V,...
            'faceColor',[0.1 0.1 0.8],'faceAlpha',.3,'edgeColor','none')
        
    end
    end
end
xlabel('x')
ylabel('y')
zlabel('z')
view(109,9.7)
%axis([0 336 0 336 0 2016])
%axis([100 250 50 200 1450 1550])

        
%% test that can be removed ??
figure
        %F = [1,2,3,4,5];
        patch('Faces',F,'Vertices',V,...
            'faceColor',[0.1 0.1 0.8],'faceAlpha',.3,'edgeColor','none')
        hold on
plot3(V(F,1),V(F,2),V(F,3),'or')
view(3)
%% try to work with a global stat variable
tic
statsAll = struct();
isa = 0;
for iz = 1 : 2016
    clear statsiz
    statsiz = stats(iz).stats;
    for ilr = 1 : length(statsiz)
        isa = isa + 1;
        statsAll(isa).z = iz;
        statsAll(isa).Area = statsiz(ilr).Area;
        statsAll(isa).Centroid = statsiz(ilr).Centroid;
        statsAll(isa).ConvexHull = statsiz(ilr).ConvexHull;
        statsAll(isa).Solidity = statsiz(ilr).Solidity;
    end
end
toc

%% identify the tracheids
for iz = 1 : size(stats,2)
    for ip = 1 : size(stats(iz).stats,1)
    pos(iz).pos(ip,1) = stats(iz).stats(ip).Centroid(1);
    pos(iz).pos(ip,2) = stats(iz).stats(ip).Centroid(2);
    pos(iz).pos(ip,3) = iz;
    end
end

maxdist = 10;
longmin = 2;

[traj,tracks]=TAN_track2d(pos,maxdist,longmin)

figure, hold on,box on, view(3)
for it = 1 : 39
    plot3(traj(it).track(:,1),traj(it).track(:,2),traj(it).track(:,3),'o')
end
%% identify the tracheids
% select a starting convexhull and propagate using the convex hulls
izStart = 1050;
iz = izStart;
listB = struct();
ib = 0;

figure
if iz >1 && iz <2016
    im2D00 = mean(im3D00(:,:,max(iz-5,1):min(iz+5,2016)),3);
    im2D72 = mean(im3D72(:,:,max(iz-5,1):min(iz+5,2016)),3);
else
    im2D00 = im3D00(:,:,iz);
    im2D72 = im3D72(:,:,iz);
end
imDiff = imsubtract(im2D00,im2D72);
imagesc(im2D00), colormap gray

% list stats at current z:
listats = find([statsAll.z]==iz);
clear a b, [a,b] = max([statsAll(1,listats).Area]);
for iilr = 1 : length(listats)
    clear V F
    ilr = listats(iilr);
    X = [statsAll(ilr).ConvexHull(:,1)];
    Y = [statsAll(ilr).ConvexHull(:,2)];
    hold on,
    if iilr == b
        fclr = [0.8 0.2 0.2];
    else
        fclr = [0.1 0.1 0.8];% face color
    end
    patch('xdata',X,'ydata',Y,...
        'faceColor',fclr,'faceAlpha',.3,'edgeColor','none')
end
%%
% select the largest region
iz = izStart
listats = find([statsAll.z]==iz)
clear a b, [a,b] = max([statsAll(1,listats).Area])
ib = ib + 1
listB(ib).z           = iz
listB(ib).statsNumber = listats(b)
listB(ib).x = statsAll(listats(b)).Centroid(1,1)
listB(ib).y = statsAll(listats(b)).Centroid(1,2)
istartRegion2kill = listats(b)
%%
% propagate UP AND DOWN
DIRz = 'up';
while iz < 2016+1
    if strcmp(DIRz , 'up')
        iz = iz + 1;
        icb = listB(ib).statsNumber(end); % icb : i current bubble
    elseif strcmp(DIRz , 'down')
        iz = iz - 1;
        icb = listB(ib).statsNumber(1); % icb : i current bubble
    end
    % list stats at current z:
    listats = find([statsAll.z]==iz);
    if ~isempty(find([statsAll.z]==iz))
        % find the one that connect most with actual bubble
        Xcb = [statsAll(icb).ConvexHull(:,1)];
        Ycb = [statsAll(icb).ConvexHull(:,2)];
        clear polarea
        for iilr =  1 : length(listats)
            ilr = listats(iilr);
            X = [statsAll(ilr).ConvexHull(:,1)];
            Y = [statsAll(ilr).ConvexHull(:,2)];
            
            poly1   = polyshape([Xcb,Ycb],'Simplify',false);
            poly2   = polyshape([X  ,Y  ],'Simplify',false);
            polyout = intersect(poly1,poly2);
            polarea(iilr) = polyout.area;
        end
        
        [a,b] = max(polarea);
    else
        a = 0;
    end
    
    if a > 5
        if strcmp(DIRz , 'up')
            listB(ib).z = [listB(ib).z,iz];
            listB(ib).x = [listB(ib).x,statsAll(listats(b)).Centroid(1,1)];
            listB(ib).y = [listB(ib).y,statsAll(listats(b)).Centroid(1,2)];
            listB(ib).statsNumber = [listB(ib).statsNumber,listats(b)];
        else
            listB(ib).z = [iz,listB(ib).z];
            listB(ib).x = [statsAll(listats(b)).Centroid(1,1),listB(ib).x];
            listB(ib).y = [statsAll(listats(b)).Centroid(1,2),listB(ib).y];
            listB(ib).statsNumber = [listats(b),listB(ib).statsNumber];
        end
        statsAll(listats(b)) = [];
    elseif strcmp(DIRz , 'up')
        DIRz = 'down';
        %fprintf('DIRz is ''down'' \n')
        iz = izStart;
    else
        %fprintf('breaking \n')
        statsAll(istartRegion2kill) = [];
    fprintf('length statsAll : %0.0f \n',length(statsAll))
        break
    end
end
%% checking with a 3D figure
ib = 5;
colors = jet(15);
figure, box on, hold on
clear X3D Y3D Z3D
X3D = [listB(ib).x];
Y3D = [listB(ib).y];
Z3D = [listB(ib).z];
plot3(X3D,Y3D,Z3D,'-o','lineWidth',4)
%axis([0 336 0 336 900 1300])
%axis([170 200 150 175 900 1300])
xlabel('x'), ylabel('y'), zlabel('z')
% add convexhull
listB(ib).x3D= [];
listB(ib).y3D= [];
listB(ib).z3D= [];
for iz = 1 : size(listB(ib).statsNumber,2)
    istat = listB(ib).statsNumber(iz);
    listB(ib).x3D = [listB(ib).x3D;[statsAll(istat).ConvexHull(:,1)]];
    listB(ib).y3D = [listB(ib).y3D;[statsAll(istat).ConvexHull(:,2)]];
    listB(ib).z3D = [listB(ib).z3D;[statsAll(istat).z*ones(length([statsAll(istat).ConvexHull(:,1)]),1)]];
end
vDT = delaunayTriangulation([listB(ib).x3D],[listB(ib).y3D],[listB(ib).z3D]);
[C,v] = convexHull(vDT);
beadSTRUCT(ib).volume = v;
hTriSurf(ib).h = trisurf(C,vDT.Points(:,1),vDT.Points(:,2),vDT.Points(:,3), ...
    'FaceColor',colors(ib,:),'faceAlpha',.5);

%% 2D renderings 
colorsP = parula(2018);
figure('defaultAxesFontSize',20)
set(gcf,'position',[681   122   960   857])
axis([0 336 0 336])
hold on, box on
for iz = 1 : 1 : 2016
    clear statsiz
    statsiz = stats(iz).stats;
    listRegions = find([statsiz.Area]>25);
    for iilr = 1 : length(listRegions)
        ilr = listRegions(iilr);
        clear V F
        X = statsiz(ilr).Centroid(1,1);
        Y = statsiz(ilr).Centroid(1,2);
        plot(X,Y,'ok','markerFaceColor',colorsP(iz,:))
    end
    title(sprintf('%0.4d',iz))
    %pause(.02)
end
xlabel('x')
ylabel('y')
%% 3D renderings :xz and yz planes
colXY = parula(337);
X = []; Y = []; Z = [];
for iz = 1 : 1 : 2016
    clear statsiz
    statsiz = stats(iz).stats;
    for il = 1 : length(statsiz)
    X = [X,statsiz(il).Centroid(1,1)];
    Y = [Y,statsiz(il).Centroid(1,2)];
    Z = [Z,iz];
    end
end
figure('defaultAxesFontSize',20)
set(gcf,'position',[681   122   450   857])
plot(X,Z,'or')

figure('defaultAxesFontSize',20)
set(gcf,'position',[681   122   450   857])
plot(Y,Z,'or')
%% try to join the points:
X = []; Y = []; Z = [];
for iz = 1 : 1 : 2016
    clear statsiz
    statsiz = stats(iz).stats;
    for il = 1 : length(statsiz)
    X = [X,statsiz(il).Centroid(1,1)];
    Y = [Y,statsiz(il).Centroid(1,2)];
    Z = [Z,iz];
    end
end

nP = length(X);
Pstatus = zeros(size(X)); % 0 / 1  not paired / paired
nb = 0;
listB = struct();
fprintf('progress: %0.0f / %0.0f \n',sum(Pstatus),length(Pstatus))
% initialize listB
iz = 1088; % 1 : 2016
clear d
idzA = find(Z==iz);
nb = length(idzA);
for ib = 1 : length(idzA)
    listB(ib).X(1) = X(idzA(ib));
    listB(ib).Y(1) = Y(idzA(ib));
    listB(ib).Z(1) = Z(idzA(ib));
    Pstatus(idzA(ib)) = 1;
end
fprintf('progress: %0.0f / %0.0f \n',sum(Pstatus),length(Pstatus))

% no prediction
for iz = 1088 : 1092%2016
    
    fprintf('iz: %0.0f nbubbles: %0.0f \n',iz,length(listB))
    % look if the bubbles already defined have a bubble at next z
    % find the bubbles at this z
    clear idzbbl idzbblLOC
    idzbbl = [];
    idzbblLOC = [];
    for ib = 1 : length(listB)
        if find([listB(ib).Z]==iz)
            idzbbl = [idzbbl,ib];
            idzbblLOC = [idzbblLOC,find([listB(ib).Z]==iz)];
        end
    end
    
    idzB = find(Z==(iz+1));
        % build map length:
        clear d
    if ~isempty(idzbbl) &&  ~isempty(idzB)
        for iA = 1 : length(idzbbl)
            xxa = listB(idzbbl(iA)).X(idzbblLOC(iA));
            yya = listB(idzbbl(iA)).Y(idzbblLOC(iA));
            for iB = 1 : length(idzB)
                xxb = X(idzB(iB));
                yyb = Y(idzB(iB));
                d(iA,iB) = sqrt((xxa-xxb)^2+(yya-yyb)^2);
            end
        end
    end
    d
    for id = 1 : length(idzbbl)
        clear a imd
        [a,imd] = min(d(id,:));
        if  a < 3
            fprintf('imd: %0.0f \n',imd)
            listB(id).X = [listB(id).X, X(idzB(imd))];
            listB(id).Y = [listB(id).Y, Y(idzB(imd))];
            listB(id).Z = [listB(id).Z, Z(idzB(imd))];
            Pstatus(idzB(imd)) = 1;
        end
    end
fprintf('progress: %0.0f / %0.0f \n',sum(Pstatus),length(Pstatus))    
    
% find the bugclose all
bxa = []; bya = [];
bxb = []; byb = [];
for iA = 1 : length(idzbbl)
    bxa = [bxa,listB(idzbbl(iA)).X(idzbblLOC(iA))];
    bya = [bya,listB(idzbbl(iA)).Y(idzbblLOC(iA))];
end
for iB = 1 : length(idzB)
    bxb = [bxb,X(idzB(iB))];
    byb = [byb,Y(idzB(iB))];
end
figure
plot(bxb,byb,'+r')
hold on
plot(bxa,bya,'ob')
axis([0 336 0 336])
pause(1)
%
%     
%     % look for other bubbles
%     clear d
%     idzA = find(Z==iz);
%     idzB = find(Z==(iz+1));
%     % build map length:
%     if ~isempty(idzA) &&  ~isempty(idzB)
%         for iA = 1 : length(idzA)
%             for iB = 1 : length(idzB)
%                 d(iA,iB) = sqrt((X(idzA(iA))-X(idzB(iB)))^2);
%             end
%         end
%     end
%     [minVal,Imin] = mink(d(:),2*length(d));
%     [row,col] = ind2sub(size(d),Imin);
%     

end

%%
bxa = []; bya = [];
bxb = []; byb = [];
for iA = 1 : length(idzbbl)
    bxa = [bxa,listB(idzbbl(iA)).X(idzbblLOC(iA))];
    bya = [bya,listB(idzbbl(iA)).Y(idzbblLOC(iA))];
end
for iB = 1 : length(idzB)
    bxb = [bxb,X(idzB(iB))];
    byb = [byb,Y(idzB(iB))];
end
figure
plot(bxb,byb,'+r')
hold on
plot(bxa,bya,'ob')
%%
figure, hold on
for ib = 1 : length(listB)
    clear X Y Z
    X3D = listB(ib).X;
    Y3D = listB(ib).Y;
    Z3D = listB(ib).Z;
    plot3(X3D,Y3D,Z3D)
end
%axis([0 336 0 336 1])
% prediction
view(3)
max(Z3D)

%% Trying stuff below
%%%%%
%%%%%
%%%%%
%%
%%%%%
%%%%%
%%%%%

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


definingWall = 'off';

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
       [xch1,ych1] = ginput(1);
       clear d
       for ich = 1 : length(channels)
           idx = find(channels(ich).z  == iz);
           xch = channels(ich).x(idx);
           ych = channels(ich).y(idx);
           d(ich) = ((xch-xch1)^2 + (ych-ych1)^2)^(1/2);
       end
       [~,ich1] = min(d);
       [xch2,ych2] = ginput(1);
       clear d
       for ich = 1 : length(channels)
           idx = find(channels(ich).z  == iz);
           xch = channels(ich).x(idx);
           ych = channels(ich).y(idx);
           d(ich) = ((xch-xch2)^2 + (ych-ych2)^2)^(1/2);
       end
       [~,ich2] = min(d);
       % go to the smallest z where they exist together
       %   find the wall
       for iw = 1 : length(walls)
           if      (ich1 == walls(iw).ch(1) && ich2 == walls(iw).ch(2)) 
                break
           elseif  (ich1 == walls(iw).ch(2) && ich2 == walls(iw).ch(1))
               break
           end
       end
       iz = walls(iw).z(1);
       definingWall = 'on';
       definingWallw = iw;
       definingWallch1 = ich1;
       definingWallch2 = ich2;
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
   switch definingWall
       case 'off'
       case 'on'
           if ishandle(h2ch)
                delete(h2ch)
           end
           axis(gca,[xAxis(1) xAxis(2) yAxis(1) yAxis(2)])
           xWch1 = channels(definingWallch1).x(walls(definingWallw).sChA(1));
           yWch1 = channels(definingWallch1).y(walls(definingWallw).sChA(1));
           xWch2 = channels(definingWallch2).x(walls(definingWallw).sChB(1));
           yWch2 = channels(definingWallch2).y(walls(definingWallw).sChB(1));
           h2ch = plot([xWch1,xWch2],[yWch1,yWch2],'og','markerFaceColor','g');
   end
   
   
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

%%
%% Old Stuffs
%%
L = bwlabeln(imDiff3D>20);
labelvolshow(L)
%%
%% split allpix_XYZ in border voxels and inner voxels
figure('defaultAxesFontSize',20), hold on, box on, view(3)
tic
% loop on voxels - for each voxel test if its 26 neighbourgs exist.
f = [];
v = [];
clear xBrdr yBrdr zBrdr nNBGRH
xBrdr = [];
yBrdr = [];
zBrdr = [];
for ivox = 1 : length(allpix_XYZ)
    if (~mod(ivox,500) == 1) || (ivox==1)
        fprintf('point: %0.0f / %0.0f \n',ivox, length(allpix_XYZ))
    end
    clear xVox yVox zVox d3D
    xVox = allpix_XYZ(ivox,1);
    yVox = allpix_XYZ(ivox,2);
    zVox = allpix_XYZ(ivox,3);
    d3D = sqrt((xVox-allpix_XYZ(:,1)).^2 + (yVox-allpix_XYZ(:,2)).^2 + (zVox-allpix_XYZ(:,3)).^2);
    nNBGRH(ivox) = length(find(d3D<sqrt(3)+0.0001));
    if nNBGRH(ivox) < 24
        color = [0.2 0.2 0.5];
        %v = xVox + [-1 0.5 0.5]/sqrt(3);
        xBrdr = [xBrdr,xVox];
        yBrdr = [yBrdr,yVox];
        zBrdr = [zBrdr,zVox];
        
%         patch('XData',xVox+[0 1 1 0],'YData',yVox+[0 0 0 0],'ZData',zVox+[0 0 1 1],'faceAlpha',0.3,'faceColor',color)
%         patch('XData',xVox+[0 1 1 0],'YData',yVox+[1 1 1 1],'ZData',zVox+[0 0 1 1],'faceAlpha',0.3,'faceColor',color)
%         
%         patch('XData',xVox+[0 0 0 0],'YData',yVox+[0 1 1 0],'ZData',zVox+[0 0 1 1],'faceAlpha',0.3,'faceColor',color)
%         patch('XData',xVox+[1 1 1 1],'YData',yVox+[0 1 1 0],'ZData',zVox+[0 0 1 1],'faceAlpha',0.3,'faceColor',color)
%         
%         patch('XData',xVox+[0 1 1 0],'YData',yVox+[0 0 1 1],'ZData',zVox+[0 0 0 0],'faceAlpha',0.3,'faceColor',color)
%         patch('XData',xVox+[0 1 1 0],'YData',yVox+[0 0 1 1],'ZData',zVox+[1 1 1 1],'faceAlpha',0.3,'faceColor',color)

    else
        color = [0.8 0.2 0.1];
    end

end
toc
%%
figure
tic
plot3(xBrdr, yBrdr, zBrdr,'ob')
toc

%%
%% segment a VoxelList 
% label matrix and show connections, 
% when cutting is finished, actualise the labels.

figure('defaultAxesFontSize',20,'position',[500 100 1200 800]), hold on, box on
% xlim([0 336])
% ylim([0 336])
xlim([160 180])
ylim([75 110])

iconn = 0;
clear listConnect
%listConnect = struct();
iz = 1499;
listCurZ = find(allpix_XYZ(:,3) == iz);
for ip = 1 : 10%length(listCurZ) 
    % find neighbourgs
    xp = allpix_XYZ(listCurZ(ip),1);
    yp = allpix_XYZ(listCurZ(ip),2);
    plot(xp,yp,'sk','markerFaceColor','b')
    % potential Neighbourgs are:
    potN(1,1) = xp + 1;
    potN(1,2) = yp + 1;
    potN(2,1) = xp - 1;
    potN(2,2) = yp + 1;
    potN(3,1) = xp + 1;
    potN(3,2) = yp - 1;
    potN(4,1) = xp - 1;
    potN(4,2) = yp - 1;
    for iN = 1 : 4
        clear d dx dy
        dx = (allpix_XYZ(listCurZ,1)-potN(iN,1)).^2;
        dy = (allpix_XYZ(listCurZ,2)-potN(iN,2)).^2;
        d = sqrt(dx+dy);
        [a,b] = min(d);
        a
        if a == 0
            iconn = iconn + 1;
            listConnect(iconn,1) = ip;
            listConnect(iconn,2) = b;
            % find the group
            listConnect(iconn,3) = 1;
        end
        
    end
    pause(.1)
end

%%

figure('defaultAxesFontSize',20,'position',[500 100 1200 800]), hold on, box on
xlim([0 336])
ylim([0 336])

% make a map of the connections between points 
conMap = struct();
labMat = struct();

clear listPCut, listPCut = [];
for iz =  1493 : 1504
    if exist('h')
        if isvalid(h)
            delete(h)
        end
    end
    if exist('hgi','var')
            delete(hgi)
    end
    % select all pixels at iz
    listCurZ = find(allpix_XYZ(:,3) == iz);
    
    h = plot(allpix_XYZ(listCurZ,1),allpix_XYZ(listCurZ,2),'sk');
    xlim([min(allpix_XYZ(listCurZ,1)) max(allpix_XYZ(listCurZ,1))])
    ylim([min(allpix_XYZ(listCurZ,2)) max(allpix_XYZ(listCurZ,2))])
    axis equal
    igi = 0 ;
    while(1)
        [x,y] = ginput(1);
        if x<min(allpix_XYZ(listCurZ,1))
            break
        end
        % find the corresponding point.
        clear d
        d = sqrt((x-allpix_XYZ(listCurZ,1)).^2 + (y-allpix_XYZ(listCurZ,2)).^2);
        [~,ip] = min(d);
        ip = listCurZ(ip);
        listPCut = [listPCut,ip];
        igi = igi + 1;
        hgi(igi) = plot(allpix_XYZ(ip,1),allpix_XYZ(ip,2),'+r');
    end
end
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


%%
function [traj,tracks]=TAN_track2d(pos,maxdist,longmin)

% nearest neighbor particle tracking algo
% pos is a structure with field .pos : pos.pos(:,1)=x1 ; pos.pos(:,2)=x2
% pos.pos(:,3)=framenumber
% frame number must an integer from 1->N
% maxdist=1; disc radius in where we are supposed to find a particle in
% next frame
% longmin : minimum length trajectory
%
% in case there are few frames bit many particles this code could be
% improved using say 100^2 vertex (only try particle in nearest vertices)
%
% example:
% load tracking_rom_positions.mat
% tracks=track2d_rom(positions,1,50);
%
% use plot_traj to display results
%
% written by r. volk 09/2014 (modified 01/2020)

tic;

tracks = pos;
for ii = 1:size(tracks,2)
    % frame number in the trajectory (from 1 ->p) for a trajectory of length p
    tracks(ii).pos(:,4) = zeros(size(tracks(ii).pos,1),1);
    % 5th column 1 if particle is free to be linked to a new trajectory, 0 if no, 2 if
    % linked to 2 or more trajectories.
    tracks(ii).pos(:,5) = ones(size(tracks(ii).pos,1),1);
    % we don't create trajectories, we only write numbers in the 4th and 5th
    % columns
end


% number of active tracks, we strat with frame 1
ind_actif = (1:size(tracks(1).pos,1));

tracks(1).pos(:,4) = (1:size(tracks(1).pos,1));
tracks(1).pos(:,5) = zeros(size(tracks(1).pos,1),1);

% number of trajectories created at this step
% will increase each time we create a new trajectory
numtraj=size(tracks(1).pos,1);

% loop over frames
%tic
for kk=2:size(tracks,2)
    % frame number we are looking at
    %numframe=kk;
    % indices of those paricles
    %ind_new=find(tracks(:,3)==numframe);

    % loop over active particles in previous frame (kk-1)
    for ll=1:length(ind_actif)
        % position of particle ll in frame kk-1
        actx = tracks(kk-1).pos(ind_actif(ll),1);
        acty = tracks(kk-1).pos(ind_actif(ll),2);

        % trajectory number of the active particle (frame kk-1)
        actnum = tracks(kk-1).pos(ind_actif(ll),4);

        % could add a tag: frame number in this trajectory
        % si tag<=2 : rien
        % si tag==3 actx=actx+vx*dt avec dt=1 ici (3 frames best estimate)
        % si tag==4 actx

        % new particle positions in frame kk
        newx = tracks(kk).pos(:,1);
        newy = tracks(kk).pos(:,2);


        % compute distance
        dist = sqrt((actx-newx).^2+(acty-newy).^2);
        % take the min
        [dmin,ind_min]=min(dist);

        % test with maxdist criterion
        if dmin < maxdist
            dispo = tracks(kk).pos(ind_min,5);

            if dispo==1
                % part is dispo=free we change dispo into 0
                tracks(kk).pos(ind_min,5) = 0;

                % we link the particle to the active particle set
                % trajectory number equal to the one of the active
                % particle
                tracks(kk).pos(ind_min,4) = actnum;

            elseif dispo==0
                % the part already linked, change dispo into 2
                % can't be linked to 2 trajectories
                tracks(kk).pos(ind_min,5) = 2;

                % and we set its trajectory number to zero
                % will be rejected at the end
                tracks(kk).pos(ind_min,4) = 0;

            end
        end
    end

        % define particles to be tracked
        % keep particles found only one time, and the non found particles
        % those will create new trajectories
        ind_actif = find(tracks(kk).pos(:,5)==0);

        % new (not found) particles are given a new trajectory number
        ind_new_part = find(tracks(kk).pos(:,5)==1);

        % if there are new particles
        if isempty(ind_new_part)==0
            % loop of new part -> increase numtraj
            for mm=1:length(ind_new_part)
                numtraj = numtraj + 1;
                tracks(kk).pos(ind_new_part(mm),4) = numtraj;
            end
        end
        ind_actif=[ind_actif;ind_new_part];
    %toc
end

% write trajectories in right order.
% reject all particles found 2 times
% keep only trajectories longer than longmin

tracks_array = cat(1,tracks.pos); %put all traj in an array
tracks_array = sortrows(tracks_array,4); %sort traj in ascendent order

tracks_array = tracks_array(tracks_array(:,4)~=0,:); %kick numtraj == 0 
fstart = zeros(length(tracks_array),1);
fstart(1) = 1; fstart(2:end) = diff(tracks_array(:,4)); %make diff -> 0 when two 2 succesive lines belong to the same traj, 1 if not
fstart = find(fstart==1); %find indices corresponding to the start of trajectories

flong = diff(fstart); %find the length of trajectories (except the last one)
last_traj = find(tracks_array(:,4)==length(fstart)); %find the length of the last trajectory
flong(end+1) = length(last_traj);

ftrackselec = find(flong>=longmin); %select traj longer than longmin

traj = struct(); %rearange in a structure
for kk = 1 :length(ftrackselec)
    ll  = flong(ftrackselec(kk));
    deb = fstart(ftrackselec(kk));
    traj(kk).track = tracks_array(deb:deb+ll-1,:);
end
end

