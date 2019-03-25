function acc_per_class = func_recognition(trainFeatures,testFeatures,trainLabels,testLabels,classifierType)
num_class = max(trainLabels);
if strcmp(classifierType,'1nn')
    % 1-NN
    distances = EuDist2(testFeatures,trainFeatures);
    [~,ind] = min(distances');
    predLabels = trainLabels(ind);
elseif strcmp(classifierType,'svm')
    % SVM
    prob = svm_classify(trainFeatures,testFeatures,trainLabels',testLabels');
    [~,predLabels] = max(prob');
elseif strcmp(classifierType,'nc')%nearest class
    distances = EuDist2(testFeatures,trainFeatures);
    classMeanDist = zeros(size(distances,1),num_class);
    for i = 1:num_class
        classMeanDist(:,i) = mean(distances(:,trainLabels==i),2);
    end
    expMatrix = exp(-classMeanDist);
    probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
    [prob,predLabels] = max(probMatrix');    
end
% calculate ACC
acc = sum(predLabels==testLabels)/length(testLabels);
for i = 1:num_class
    acc_per_class(i) = sum((predLabels == testLabels).*(testLabels==i))/sum(testLabels==i);
end
fprintf('Acc:%0.3f,Mean acc per class: %0.3f\n', acc, mean(acc_per_class));
