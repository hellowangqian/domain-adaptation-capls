% =====================================================================
% Code for conference paper:
% Qian Wang, Penghui Bu, Toby Breckon, Unifying Unsupervised Domain
% Adaptation and Zero-Shot Visual Recognition, IJCNN 2019
% By Qian Wang, qian.wang173@hotmail.com
% =====================================================================
function [acc, acc_per_class] = DA_LPP_SP(domainS_features,domainS_labels,domainT_features,domainT_labels,d,T)
num_iter = T;
options.NeighborMode='KNN';
options.WeightMode = 'HeatKernel';
options.k = 30;
options.t = 1;
options.ReducedDim = d;
options.alpha = 1;
num_class = length(unique(domainS_labels));
W_all = zeros(size(domainS_features,1)+size(domainT_features,1));
W_s = constructW1(domainS_labels);
W = W_all;
W(1:size(W_s,1),1:size(W_s,2)) =  W_s;
% looping
p = 1;
fprintf('d=%d\n',options.ReducedDim);
for iter = 1:num_iter
    P = LPP([domainS_features;domainT_features],W,options);
    %P = LPP(domainS_features,W_s,options);
    domainS_proj = domainS_features*P;
    domainT_proj = domainT_features*P;
    proj_mean = mean([domainS_proj;domainT_proj]);
    domainS_proj = domainS_proj - repmat(proj_mean,[size(domainS_proj,1) 1 ]);
    domainT_proj = domainT_proj - repmat(proj_mean,[size(domainT_proj,1) 1 ]);
    domainS_proj = L2Norm(domainS_proj);
    domainT_proj = L2Norm(domainT_proj);
    %forPlot{1} = [forPlot{1};domainS_proj(domainS_labels<11,:)];
    %forPlot{2} = [forPlot{2};domainT_proj(domainT_labels<11,:)];
    %classTh=21;
    %my_tsne(domainS_proj(domainS_labels<classTh,:),domainT_proj(domainT_labels<classTh,:),domainS_labels(domainS_labels<classTh),domainT_labels(domainT_labels<classTh),classTh);
    %% distance to class means
    classMeans = zeros(num_class,options.ReducedDim);
    for i = 1:num_class
        classMeans(i,:) = mean(domainS_proj(domainS_labels==i,:));
    end
    classMeans = L2Norm(classMeans);
    distClassMeans = EuDist2(domainT_proj,classMeans);
    targetClusterMeans = vgg_kmeans(double(domainT_proj'), num_class, classMeans')';
    targetClusterMeans = L2Norm(targetClusterMeans);
    distClusterMeans = EuDist2(domainT_proj,targetClusterMeans);
    expMatrix = exp(-distClassMeans);
    expMatrix2 = exp(-distClusterMeans);
    probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
    probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 num_class]);
    probMatrix = max(probMatrix,probMatrix2);
    %probMatrix = probMatrix2;
    [prob,predLabels] = max(probMatrix');
    p=1-iter/(num_iter-1);
    p = max(p,0);
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
    W = constructW1([domainS_labels,pseudoLabels]);
    %% calculate ACC
    acc(iter) = sum(predLabels==domainT_labels)/length(domainT_labels);
    for i = 1:num_class
        acc_per_class(iter,i) = sum((predLabels == domainT_labels).*(domainT_labels==i))/sum(domainT_labels==i);
    end
    fprintf('Iteration=%d/%d, Acc:%0.3f,Mean acc per class: %0.3f\n', iter,num_iter, acc(iter), mean(acc_per_class(iter,:)));
    if sum(trustable)>=length(prob)
        break;
    end
end
