clc
clear all
close all

%% detection yolo

% % % load training label data

load aaa

% input size.
imageSize = [224 224 3];

%no of class
numClasses = 1;

anchorBoxes = [
    43 59
    18 22
    23 29
    84 109
];


%residual network 

base = resnet50;

inputlayer=base.Layers(1)

middle =base.Layers(2:174)

finallayer=base.Layers(175:end)

baseNetwork=[inputlayer
    
               middle
               
               finallayer]

% Specify the feature extraction layer.

featureLayer = 'activation_40_relu';

%% the YOLO v2 object detection network. 

lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,base,featureLayer);

options = trainingOptions('sgdm', ...
        'MiniBatchSize', 16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',30,...
        'CheckpointPath', tempdir, ...
        'Shuffle','every-epoch');    
% % % %     
vehicleDataset=gTruth;
[detector,info] = trainYOLOv2ObjectDetector(vehicleDataset,lgraph,options);


%%% Load trained network 

%%%load zzzz


% % Read a test image.
% I = imread('detectcars.png');
% 
% I=imresize(I,[224 224]);
% 
% % Run the detector.
% [bboxes,scores] = detect(detector,I);
% 
% % Annotate detections in the image.
% I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
% imshow(I)


%% CLASSIFICATION 

% % % set directory 
matlabroot='C:\Users\mebin\Desktop\2019new\vechicle\Dataset'

% % set path
DatasetPath = fullfile(matlabroot);
Data = imageDatastore(DatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
% % count the class
CountLabel = Data.countEachLabel;



trainData=Data;

%% Define the Network Layers
% Define the convolutional neural network architecture. 
layers = [imageInputLayer([224 224 3])
    
          convolution2dLayer(5,20)
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          
          convolution2dLayer(5,20)
          reluLayer
          maxPooling2dLayer(2,'Stride',2)

          fullyConnectedLayer(2)
          softmaxLayer
          
          classificationLayer()];

options = trainingOptions('sgdm','MaxEpochs',20, ...
	'InitialLearnRate',0.0001);  

% % train the network
convnet = trainNetwork(trainData,layers,options);



% train1=detector;

% % Input video 

v = VideoReader('Traffic2.asf')


myVideo = VideoWriter('Output_ACF1.avi');
myVideo.Quality = 50;
myVideo.FrameRate = 15; 
k=0;

% cd data2
% cd data
  for i=1:50
     str=int2str(i);
     

 
% %     
videoFrame  = read(v,i);
I=videoFrame;

I=imresize(I,[224 224]);

% Run the detector.
[bboxes,scores,label] = detect(detector,I);

% label
% 
% % 
% % imwrite(videoFrame,str)
% 
% if isempty(bbox)
%     detect1=a1;
% else
%     
%  detect1 = insertShape(I, 'Rectangle', bboxes);

% % end
% figure(1)
% imshow(detect1)
% % pause(0.5)


%   
   all=[];
   lab=[];
   kk=0;
 for ii=1: size(bboxes,1)
     kk=kk+1;
%      
%      se=int2str(kk)
%  str1=strcat(str,se,'.png')
cr=imcrop(I,bboxes(ii,:));

cr=imresize(cr,[224 224]);
      
out = classify(convnet,cr)

    
%     
%     
%  
% %     imwrite(cr,str1)
%     
%     out=round(rand(1,1));
 
    
    if out=='car'
        aaa='car';
    elseif out=='truck'
        aaa='truck';
    elseif out==2
        aaa='other';
    end
    
    if out=='car';   
        lab=[lab ;{aaa}];
    else
        lab=[lab; {aaa}];
        
    end
     
 end


label_str = cell(size(lab,1),1);
conf_val = lab;
for iii=1:size(lab,1)
    label_str{iii} = conf_val{iii};
end
% Set the position for the rectangles as [x y width height].

 position = bboxes;
% Insert the labels.

RGB = insertObjectAnnotation(I,'rectangle',position(1:size(position,1),:),label_str(1:size(position,1)),...
    'TextBoxOpacity',0.9,'FontSize',18);
% Display the annotated image.
% 
% figure
% imshow(RGB)
% title('Annotated chips');


% elseif outt==2
% 
% detect1 = insertShape(I, 'Rectangle', bboxes(ii,:));
%     end
% 
%     all=[all outt]
% end    
%   
figure(1),imshow(RGB);

open(myVideo)
    writeVideo(myVideo,(RGB))
 
  end
  
  
  vv= VideoReader('Output_ACF1.avi')

  % cd labell
%    fps = 0

for zz=1:50
      % Read frames from video
      im = read(vv,zz);      
      im = imresize(im,[300,600]);
      figure(2),imshow(im)
   end
   
%    
% step(videoPlayer, videoFrame);





