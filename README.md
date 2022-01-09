# Traffic-Light-Detection-Using-Yolo-v2-Model


### Load newfinal.m to matlab
### place all other files in current folder(Left side of matlab window)
### For Bounding Boxes, Type 'imageLabeller' in command window
### In imageLabeler window put Bounding Boxes to all your images
### And give bb.mat as your file name.


## Training Phase 

### Step 1: Take input dataset. 
### Step 2: Divide the dataset into 60:20:20 ratio for training , validating and testing respectively. 
### Step 3:Image label-er loads all the images and directs them further to be put in bounding box / anchor box. 
### Step 4: All the images with labels and bounding box are saved and imported in the MATLAB workspace. 
### Step 5: Resize the images to 224x224. 
### Step 6:Specify the number of classes - 3 i.e.’Red’,’Green’ and ’Yellow’.
### Step 7:Use the residual learning framework to ease the training of these networks.
### Step 8:Load the YOLO v2 object detection network. 
### Step 9: The model is trained for the given dataset


## Testing Phase 

### Step 1:Input the test image.
### Step 2: Select the particular piece of code and evaluate the section for testing different images. 
### Step 3:The trained model detects the Traffic light and gives labelled output with state of the Traffic Light detected accurately. 


### [You can download dataset here](https://drive.google.com/drive/u/0/folders/1QXR2Kc5YqCrJliHcloDV9PrjCH2TqZYU)
