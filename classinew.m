
%% CLASSIFICATION 
clc
clear all
close all

% % % set directory 
matlabroot='C:\Users\mebin\Desktop\2019new\vechicle\vechicledetect\fulldata'

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

          fullyConnectedLayer(4)
          softmaxLayer
          
          classificationLayer()];

options = trainingOptions('sgdm','MaxEpochs',10, ...
	'InitialLearnRate',0.0001);  

% % train the network
convnet = trainNetwork(trainData,layers,options);


load vh.mat
detector=vechi;

% % Input video 

v = VideoReader('8k.mp4')


myVideo = VideoWriter('Output_ACF1.avi');
myVideo.Quality = 50;
myVideo.FrameRate = 15; 
k=0;

% cd data2
% cd data
  for i=1:4:1000
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

figure(1),imshow(cr)
title(char(out))

 end
    
  end



