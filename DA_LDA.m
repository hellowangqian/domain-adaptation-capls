% =====================================================================
% Code for conference paper:
% Qian Wang, Penghui Bu, Toby Breckon, Unifying Unsupervised Domain
% Adaptation and Zero-Shot Visual Recognition, IJCNN 2019
% By Qian Wang, qian.wang173@hotmail.com
% =====================================================================
function acc_per_class = DA_LDA(domainS_features,domainS_labels,domainT_features,domainT_labels)
num_iter = 20;
num_class = length(unique(domainS_labels));
trainLabels = domainS_labels;
trainFeatures = domainS_features;
options.gnd = trainLabels';
% looping
for iter = 1:num_iter
    [P,~] = LDA(trainLabels',options,double(trainFeatures));
    %P = LPP(domainS_features,W_s,options);
    domainS_proj = domainS_features*P;
    domainT_proj = domainT_features*P;
    proj_mean = mean([domainS_proj;domainT_proj]);
    domainS_proj = domainS_proj - repmat(proj_mean,[size(domainS_proj,1) 1 ]);
    domainT_proj = domainT_proj - repmat(proj_mean,[size(domainT_proj,1) 1 ]);
    domainS_proj = L2Norm(domainS_proj);
    domainT_proj = L2Norm(domainT_proj);
    distances = EuDist2(domainT_proj,domainS_proj);
    %% distance to class means
    classMeans = zeros(num_class,size(domainS_proj,2));
    for i = 1:num_class
        classMeans(i,:) = mean(domainS_proj(domainS_labels==i,:));
    end
    classMeans = L2Norm(classMeans);
    distClassMeans = EuDist2(domainT_proj,classMeans);
    expMatrix = exp(-distClassMeans);
    probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
    [prob,predLabels] = max(probMatrix');
    %%
    p=1-iter/num_iter;
    [sortedProb,index] = sort(prob);
    sortedPredLabels = predLabels(index);
    trustable = zeros(1,length(prob));
    for i = 1:num_class
        thisClassProb = sortedProb(sortedPredLabels==i);
        if length(thisClassProb)>0
            trustable = trustable+ (prob>thisClassProb(floor(length(thisClassProb)*p)+1)).*(predLabels==i);
        end
    end
    pseudoLabels = predLabels;
    pseudoLabels(~trustable) = -1;
    trainFeatures = [domainS_features;domainT_features(logical(trustable),:)];
    trainLabels = [domainS_labels,pseudoLabels(logical(trustable))];
    options.gnd = trainLabels';
    %% calculate ACC
    acc = sum(predLabels==domainT_labels)/length(domainT_labels);
    for i = 1:num_class
        acc_per_class(i) = sum((predLabels == domainT_labels).*(domainT_labels==i))/sum(domainT_labels==i);
    end
    fprintf('Iteration=%d, Acc:%0.3f,Mean acc per class: %0.3f\n', iter, acc, mean(acc_per_class));
    if sum(trustable)>=length(prob)
        break;
    end
end