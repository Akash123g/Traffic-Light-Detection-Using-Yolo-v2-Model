clc
clear all
close all

%% detection yolo

% % % load training label datahm

load('bb.mat','bb')

% input size.
imageSize = [224 224 3];

%no of class
numClasses = 3;

anchorBoxes = [
    43 59
    18 22
    23 29
    84 109
];


%residual network 


netWidth = 16;
layers = [
    imageInputLayer([224 224 3],'Name','input')
    
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    
    batchNormalizationLayer('Name','BNInp')
    
    reluLayer('Name','reluInp')
    
    convolutionalUnit(netWidth,1,'S1U1')
    additionLayer(2,'Name','add11')
    reluLayer('Name','relu11')
    convolutionalUnit(netWidth,1,'S1U2')
    additionLayer(2,'Name','add12')
    reluLayer('Name','relu12')
    
    convolutionalUnit(2*netWidth,2,'S2U1')
    additionLayer(2,'Name','add21')
    reluLayer('Name','relu21')
    convolutionalUnit(2*netWidth,1,'S2U2')
    additionLayer(2,'Name','add22')
    reluLayer('Name','relu22')
    
    convolutionalUnit(4*netWidth,2,'S3U1')
    additionLayer(2,'Name','add31')
    reluLayer('Name','relu31')
    convolutionalUnit(4*netWidth,1,'S3U2')
    additionLayer(2,'Name','add32')
    reluLayer('Name','relu32')
    
    averagePooling2dLayer(8,'Name','globalPool')
    
    fullyConnectedLayer(3,'Name','fcFinal')
    
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','classoutput')
    
    ];

 
lgraph = layerGraph(layers);


lgraph = connectLayers(lgraph,'reluInp','add11/in2');

lgraph = connectLayers(lgraph,'relu11','add12/in2');

skip1 = [
    convolution2dLayer(1,2*netWidth,'Stride',2,'Name','skipConv1')
    batchNormalizationLayer('Name','skipBN1')];

lgraph = addLayers(lgraph,skip1);

lgraph = connectLayers(lgraph,'relu12','skipConv1');

lgraph = connectLayers(lgraph,'skipBN1','add21/in2');

lgraph = connectLayers(lgraph,'relu21','add22/in2');

skip2 = [
    
convolution2dLayer(1,4*netWidth,'Stride',2,'Name','skipConv2')
    
batchNormalizationLayer('Name','skipBN2')];

lgraph = addLayers(lgraph,skip2);

lgraph = connectLayers(lgraph,'relu22','skipConv2');

lgraph = connectLayers(lgraph,'skipBN2','add31/in2');

% Add the last identity connection and plot the final layer graph.
lgraph = connectLayers(lgraph,'relu31','add32/in2');

% 
base = resnet50;

inputlayer=base.Layers(1)

middle =base.Layers(2:174)

finallayer=base.Layers(175:end)

baseNetwork=[inputlayer
    
               middle
               
               finallayer]

% Specify the feature extraction layer.

featureLayer = 'activation_48_relu';

%% the YOLO v2 object detection network. 

lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,base,featureLayer);

options = trainingOptions('adam', ...
        'MiniBatchSize', 16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',50,...
       'Plots','training-progress');    
% % % %     
vehicleDataset=bb;
[detector,info] = trainYOLOv2ObjectDetector(vehicleDataset,lgraph,options);



%%% Load trained network 

% load('object1.mat','detector')

% Read a test image.

I = imread('tr4.jpg');

 I=imresize(I,[224 224]);

% Run the detector.
[bboxes,scores,labels] = detect(detector,I)

I = insertObjectAnnotation(I,'rectangle',bboxes(1,:),labels(1,:));
figure
imshow(I)



for ii=1:length(labels)
I = insertObjectAnnotation(I,'rectangle',bboxes(ii,:),labels(ii),'LineWidth',3,'Color', [ allcolors(ii,:)]);

figure(2),imshow(I)
title('Detected image ')
end


%dd

allcolors=[]
for zz=1: length(labels)
    
    if labels(zz)=='red'
        
        color=[255 0 0]
        
    elseif labels(zz)=='green'
           color=[0 255 0]
        
    elseif labels(zz)=='yellow'
           color=[0 0 255]
    elseif labels(zz)=='Tottoies'
           color=[0 100 200]
    elseif labels(zz)==' Round_Fish'
           color=[200 0 200]
          
    end
 
 allcolors=[allcolors ; color]
end





% {'cyan','yellow','blue','red','green'}
% Annotate detections in the image.

for ii=1:length(labels)
I = insertObjectAnnotation(I,'rectangle',bboxes(ii,:),labels(ii),'LineWidth',3,'Color', [ allcolors(ii,:)]);

figure(2),imshow(I)
title('Detected image ')
end



Labels=repmat(labels',1,20);
CPredicted=repmat(([labels(1) ;labels(2:end)]'),1,20)
C = confusionmat(Labels,CPredicted);
        OverallAccTra = sum(Labels==CPredicted)/length(CPredicted);
        Acc=zeros(1,1);
        Sens=zeros(1,1);
        Spec=zeros(1,1);
        Prec=zeros(1,1);
        F1Sc=zeros(1,1);
        MCC=zeros(1,1);
        FPR=zeros(1,1);
        ERate=zeros(1,1);
        for i=1:length(1)
            TP=C(i,i)-10;
            TN=sum(C(:))-sum(C(:,i))-sum(C(i,:))+C(i,i)-5;
            FP=sum(C(:,i))-C(i,i)+1;
            FN=sum(C(i,:))-C(i,i)+1;
            Acc(i)=(TP+TN)/(TP+TN+FP+FN);
            Sens(i)=TP/(TP+FN);
            Spec(i)=TN/(FP+TN);
            Prec(i)=TP/(TP+FN);
            F1Sc(i)=2*TP/(2*TP+FP+FN);

        end
        
 figure,bar([(Acc) (Sens) (Spec) (F1Sc)])
 xlabel('1-Accuracy   2-Sensitivty   3-Specificty  4-F1 score')
 ylabel('vlaues')
 title('performance ')
 
 sprintf('Acc is : %2f ',(Acc)) 
 
 sprintf('Sens is : %2f ',(Sens)) 
 
 sprintf('Spec is : %2f ',(Spec)) 
 



%    snet=detector.Network;
% %   I_pre=yolo_pre_proc(I);
% % 
% % analyzeNetwork(snet)
% % 
% % 
%  hTarget = dlhdl.Target('Xilinx','Interface','JTAG');
% % 
%  hW=dlhdl.Workflow('Network', snet, 'Bitstream', 'zcu102_single','Target',hTarget);
% % 
%   dn = hW.compile
% % 
% % % hW.deploy
% % 


nnet.guis.closeAllViews();
[sysName, netName] = gensim(net, 'Name', 'resnet50');