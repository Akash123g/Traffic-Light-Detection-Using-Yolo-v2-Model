camera = webcam; % Connect to the camera
net = alexnet;   % Load the neural network

hdlsetuptoolpath('ToolName','Xilinx Vivado','ToolPath',...
 'E:\Vivado\2018.1\bin\vivado.bat');

hT = dlhdl.Target('Xilinx');
hW = dlhdl.Workflow('Network',net,'Bitstream','zcu102_single','Target',hT);
hW.deploy;

while true
    im = snapshot(camera);       % Take a picture
    image(im);                   % Show the picture
    im = imresize(im,[227 227]); % Resize the picture for alexnet
    [prediction, speed] = hW.predict(single(im),'Profile','on');
    [val, idx] = max(prediction);
    label = net.Layers(end).ClassNames{idx}; % Classify the image
    title(char(label));          % Show the class label
    drawnow
end
   
