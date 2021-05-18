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
%%
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
tic
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
toc

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

%% preload images - long time scale ( 20 minutes )
tic
it = 72;
file00 = strcat(nameFolder,nameExpe,sprintf('_%0.4d.tif',it));
file00 = 'C:\Users\Lenovo\Jottacloud\RECHERCHE\Projets\02_ESRF\data\ESRF_full_sequences\pinus02-w-P2\pinus02-w-P2_0072.tif';
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
toc

it = 0;
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

%% im subtract and regions props of 2D images

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
for iz =  1 : 10 : 2016
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

% select the largest region
ib = ib + 1;
listB(ib).z           = iz;
listB(ib).statsNumber = listats(b);
listB(ib).x = statsAll(listats(b)).Centroid(1,1);
listB(ib).y = statsAll(listats(b)).Centroid(1,2);

% propagate UP AND DOWN
DIRz = 'up';
while iz<2016
    DIRz
    if strcmp(DIRz , 'up')
        iz = iz + 10
    elseif strcmp(DIRz , 'down')
        iz = iz - 10
    end
    % list stats at current z:
    listats = find([statsAll.z]==iz);
    % find the one that connect most with actual bubble
    icb = listB(ib).statsNumber(end); % icb : i current bubble
    Xcb = [statsAll(icb).ConvexHull(:,1)];
    Ycb = [statsAll(icb).ConvexHull(:,2)];
    clear polarea
    for iilr =  1 : length(listats)
        ilr = listats(iilr);
        X = [statsAll(ilr).ConvexHull(:,1)];
        Y = [statsAll(ilr).ConvexHull(:,2)];
        
        poly1   = polyshape(Xcb,Ycb);
        poly2   = polyshape(X  ,Y  );
        polyout = intersect(poly1,poly2);
        polarea(iilr) = polyout.area;
    end
    
    [a,b] = max(polarea);
    if a > 5
        listB(ib).z = [listB(ib).z,iz];
        listB(ib).x = [listB(ib).x,statsAll(listats(b)).Centroid(1,1)];
        listB(ib).y = [listB(ib).y,statsAll(listats(b)).Centroid(1,2)];
        listB(ib).statsNumber = [listB(ib).statsNumber,listats(b)];
    elseif strcmp(DIRz , 'up')
        DIRz = 'down';
        iz = izStart;
    else
        break
    end
end
%% checking with a 3D figure
figure, box on
clear X3D Y3D Z3D
X3D = [listB(ib).x];
Y3D = [listB(ib).y];
Z3D = [listB(ib).z];
plot3(X3D,Y3D,Z3D,'-o','lineWidth',4)
%axis([0 336 0 336 900 1300])
%axis([170 200 150 175 900 1300])
xlabel('x'), ylabel('y'), zlabel('z')
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


