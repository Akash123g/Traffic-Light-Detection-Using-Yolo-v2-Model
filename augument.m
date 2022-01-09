clc
clear all
close all
g
v = VideoReader('7k.mp4');

cd data5
% cd data
  for i=1:2:1000
      str=int2str(i)
% %     
    str1=strcat(str,'.png')
videoFrame  = read(v,i);

videoFrame1 =imresize(videoFrame ,[224 224]);

figure(2),imshow(videoFrame1)
% 
imwrite(videoFrame1,str1)
 
  end