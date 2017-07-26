function feat = feature_extraction(img)
% Output should be a fixed length vector [1*dimension] for a single image. 
% Please do NOT change the interface.
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);
R = [mean(mean(R(1:floor(end/2),:))) mean(mean(R(floor(end/2)+1:end,:)))];
G = [mean(mean(G(1:floor(end/2),:))) mean(mean(G(floor(end/2)+1:end,:)))];
B = [mean(mean(B(1:floor(end/2),:))) mean(mean(B(floor(end/2)+1:end,:)))];
TR = graythresh(R);
TG = graythresh(G);
TB = graythresh(B);
meanColors = [R G B TR TG TB]/255;
model = load('model.mat');
C = model.C;
histFeatures = model.histFeatures;
testHist = zeros(1,size(C,1));
img = rgb2gray(img);
points = detectSURFFeatures(img);
[features,validPoints] = extractFeatures(img, points.selectStrongest(300),'SURFSize',128);
for k=1:size(features,1)
    for r=1:size(C,1)
        a(r) = sqrt(sum((features(k,:) - C(r,:)) .^ 2));
    end
    [M,I] = min(a);
    testHist(I) = testHist(I)+1;
end

feat = testHist/sum(testHist);
feat = [feat meanColors];
end