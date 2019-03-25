function y=svm_classify(trainData,testData,trainLabel,testLabel)
numLabels=max(testLabel);
numTest=size(testLabel,1);
trainData = double(trainData);
testData = double(testData);
trainLabel = double(trainLabel);
testLabel = double(testLabel);
%% PCA
% fprintf('PCA...\n');
% [coef,score,latent]=princomp(trainData,'econ');
% testPCA=bsxfun(@minus,testData,mean(testData,1))*coef(:,1:3000);
%%
% trainData=score(:,1:3000);
% testData=testPCA;
%# train one-against-all models
%% # get probability estimates of test instances using each model
prob = zeros(numTest,numLabels);
for k=1:numLabels
    %fprintf('train=%d\n',k);
    model = svmtrain(double(trainLabel==k),trainData, '-t 0 -b 1 -q');
    [~,~,p] = svmpredict(double(testLabel==k),testData, model, '-b 1');
    prob(:,k) = p(:,model.Label==1);    %# probability of class==k
end
%# predict the class with the highest probability
[~,pred] = max(prob,[],2);
acc = sum(pred == testLabel) ./ numel(testLabel);    %# accuracy
%C = confusionmat(testLabel, pred)                   %# confusion matrix
y=prob;
end