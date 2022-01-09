clc
clear all
close all


cd dataset
flist = dir('.\truck\*.png');

cd 

for I = 1:100
%%% Read Input Image
    
    Img = imresize(imread(['.\truck\' flist(I).name]),[224 224]);
% cd data
% ghgh
% ghg
%       str=int2str(i)
% % %     
%     str=strcat(str,'.png')
% videoFrame  = imread(flist(I).name);

videoFrame =imresize(Img ,[224 224]);

 imwrite(videoFrame,flist(I).name)
 
  end