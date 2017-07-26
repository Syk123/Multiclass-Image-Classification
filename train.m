% You can change anything you want in this script.
% It is provided just for your convenience.

% K is number of clusters
% h is the histogram for a single category
% histFeatrues contains 1800 histograms, one for each image


clear; clc; close all;

K = 30;
img_path = './train/';
class_num = 30;
img_per_class = 60;
img_num = class_num .* img_per_class;

folder_dir = dir(img_path);
label_train = zeros(img_num,1);
allFeatures = [];
nFeatList = zeros(img_num,1);
color_feat = zeros(img_num,9);
for i = 1:length(folder_dir)-2
    
    img_dir = dir([img_path,folder_dir(i+2).name,'/*.JPG']);
    if isempty(img_dir)
        img_dir = dir([img_path,folder_dir(i+2).name,'/*.BMP']);
    end
    
    label_train((i-1)*img_per_class+1:i*img_per_class) = i;
    for j = 1:length(img_dir)        
        img = imread([img_path,folder_dir(i+2).name,'/',img_dir(j).name]);
        R = img(:,:,1);
        G = img(:,:,2);
        B = img(:,:,3);
        TR = graythresh(R);
        TG = graythresh(G);
        TB = graythresh(B);
        R = [mean(mean(R(1:floor(end/2),:))) mean(mean(R(floor(end/2)+1:end,:)))];
        G = [mean(mean(G(1:floor(end/2),:))) mean(mean(G(floor(end/2)+1:end,:)))];
        B = [mean(mean(B(1:floor(end/2),:))) mean(mean(B(floor(end/2)+1:end,:)))];
        meanColors = [R G B TR TG TB]/255;
        color_feat((i-1)*img_per_class+j,:) = meanColors;
        img = rgb2gray(img);

        points = detectSURFFeatures(img);
        [feat,validPoints] = extractFeatures(img, points.selectStrongest(300),'SURFSize',128);
        allFeatures = [allFeatures; feat];
        nFeatList((i-1)*img_per_class+j) = size(feat,1);
            
    end
     
end

[idx, C] = kmeans(allFeatures, K,'MaxIter',1500,'Distance','cosine');


histFeatures = [];
head = 0;

for i=1:1800
    h=hist(idx(head+1:head+nFeatList(i)), 1:K);
    head = head+nFeatList(i);
    histFeatures = [histFeatures; h];
end


for i=1:size(histFeatures,1)
    histFeatures(i,:) = histFeatures(i,:)/sum(histFeatures(i,:));
end
ni = zeros(1,size(histFeatures,2));

for i=1:size(histFeatures,2)
    ni(i) = nnz(histFeatures(:,i));
end

histFeatures = cat(2,histFeatures, color_feat);
save('model.mat','histFeatures','C');