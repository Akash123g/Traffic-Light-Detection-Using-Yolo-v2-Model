clc
clear all
close all

v = VideoReader('11.mp4');


myVideo = VideoWriter('Output_ACF2.avi');
myVideo.Quality = 50;
myVideo.FrameRate = 15; 
k=0;


  for i=500:800
     str=int2str(i);
     

 i
% %     
videoFrame  = read(v,i);
I=videoFrame;

% I=imresize(I,[224 224]);

[m n c]=size(I);

II=imcrop(I,[500 300 n 300]);

II=imresize(II,[224 224]);

figure(1),imshow(II)

  end
  
  
% figure(1),imshow(RGB);

open(myVideo)
 writeVideo(myVideo,(II))
    
    