function predict_label = your_kNN(feat)
% Output should be a fixed length vector [num of img, 1]. 
% Please do NOT change the interface.

%k is the number of nearest neighbor considered
k = 11;
class_num = 30;
img_per_class = 20;
img_num = class_num .* img_per_class;
predict_label = zeros(img_num,1);

model = load('model.mat');
histFeatures = model.histFeatures;
dis = zeros(size(histFeatures,1),1);

for i=1:size(feat,1)
    imgHist = feat(i,:);
    for j=1:size(histFeatures,1)
        dis(j) = sqrt(sum((histFeatures(j,:) - imgHist) .^ 2));
    end
    [disSorted,idx]=sort(dis);
    %predict_label(i) = floor(idx(1)/60)+1;
    idx = idx(1:k);
    
    classes = floor(idx/60)+1;
    
    classHist = hist(classes, min(classes):max(classes));
    [m1, i1] = max(classHist);
    predict_label(i) = min(classes)+i1-1;

end
