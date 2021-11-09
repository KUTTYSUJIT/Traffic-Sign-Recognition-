% Clear the command window
clear,clc,close all


% Get training images
pathToImages = 'C:/Users/sujit/Desktop/Final Project/Final Data';
trafficds = imageDatastore(pathToImages,"IncludeSubfolders",true,"LabelSource","foldernames");
trafficds


% % Resizing the images as per the input size of Pre-treined GoogleNet

srcFiles1 = dir('C:/Users/sujit/Desktop/Final Project/Final Data/No entry/*.png'); 
srcFiles2 = dir('C:/Users/sujit/Desktop/Final Project/Final Data/No horn/*.png');
srcFiles3 = dir('C:/Users/sujit/Desktop/Final Project/Final Data/U turn/*.png');
srcFiles4 = dir('C:/Users/sujit/Desktop/Final Project/Final Data/Roundabout/*.png');
srcFiles5 = dir('C:/Users/sujit/Desktop/Final Project/Final Data/Speed Limit/*.png');
srcFiles6 = dir('C:/Users/sujit/Desktop/Final Project/Final Data/stop/*.png');
srcFiles7 = dir('C:/Users/sujit/Desktop/Final Project/Final Data/Yield/*.png');
srcFiles8 = dir('C:/Users/sujit/Desktop/Final Project/Final Data/Construction/*.png');


% % Storing path for each categories of road sign

path1 = 'C:/Users/sujit/Desktop/Final Project/Final Data/No entry/';
path2 = 'C:/Users/sujit/Desktop/Final Project/Final Data/No horn/';
path3 = 'C:/Users/sujit/Desktop/Final Project/Final Data/U turn/';
path4 = 'C:/Users/sujit/Desktop/Final Project/Final Data/Roundabout/';
path5 = 'C:/Users/sujit/Desktop/Final Project/Final Data/Speed Limit/';
path6 = 'C:/Users/sujit/Desktop/Final Project/Final Data/stop/';
path7 = 'C:/Users/sujit/Desktop/Final Project/Final Data/Yield/';
path8 = 'C:/Users/sujit/Desktop/Final Project/Final Data/Construction/';
% 

% Calling image-resize function for each category of road-sign
imgresize(path1,srcFiles1)
imgresize(path2,srcFiles2)
imgresize(path3,srcFiles3)
imgresize(path4,srcFiles4)
imgresize(path5,srcFiles5)
imgresize(path6,srcFiles6)
imgresize(path7,srcFiles7)
imgresize(path8,srcFiles8)


% Create a network by modifying GoogLeNet
% Get the layers from GoogLeNet

net = googlenet;
ly = net.Layers;

inlayer = ly(1);
insz = inlayer.InputSize;
insz    


% Determine the number of distinct traffic signs
numClasses = numel(categories(trafficds.Labels));



% % Split into training and testing sets
[trainImgs,testImgs,validation] = splitEachLabel(trafficds,0.7,0.15);

% % Extract the layer-graph of the GoogleNet Network
lgraph = layerGraph(net)
 
% % Modify the classification and output layers
newFc = fullyConnectedLayer(8,"Name","new_fc")
lgraph = replaceLayer(lgraph,"loss3-classifier",newFc)
newOut = classificationLayer("Name","new_out")
lgraph = replaceLayer(lgraph,"output",newOut)


% % Set training algorithm options
% % Lower the learning rate for transfer learning
% % Change the hyperparameters in the below options

options = trainingOptions("sgdm","InitialLearnRate", 0.001,'MaxEpochs', 30,'Plots','training-progress');

% 
% % Perform training
[trafficnet,info] = trainNetwork(trainImgs, lgraph, options);

% 
% % Use the trained network to classify test images
testpreds = classify(trafficnet,testImgs);

% 
% % Evaluate the results
% % Calculate the accuracy
% 
nnz(testpreds == testImgs.Labels)/numel(testpreds)
% 
% % Visualize the confusion matrix
confusionchart(testImgs.Labels,testpreds,'RowSummary','row-normalized','ColumnSummary','column-normalized');


testActual = testImgs.Labels;

% Finding Misclassified values
idxWrong = find(testpreds ~= testActual);

% Displaying Mis-classified Images
a=1;
b=length(idxWrong);
i=1;
for j=1:a
    for k=1:b
        img = readimage(testImgs,idxWrong(i));
        subplot(a,b,i)
        imshow(img);
        i=i+1;
    end
end

