% =====================================================================
% Code for conference paper:
% Qian Wang, Penghui Bu, Toby Breckon, Unifying Unsupervised Domain
% Adaptation and Zero-Shot Visual Recognition, IJCNN 2019
% By Qian Wang, qian.wang173@hotmail.com
% =====================================================================
function acc_per_class = DA_LPP_noSel(domainS_features,domainS_labels,domainT_features,domainT_labels)
num_iter = 10;
options.NeighborMode='KNN';
options.WeightMode = 'HeatKernel';
options.k = 30;
options.t = 1;
options.ReducedDim = 128;
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
    domainS_proj = domainS_features*P;
    domainT_proj = domainT_features*P;
    proj_mean = mean([domainS_proj;domainT_proj]);
    domainS_proj = domainS_proj - repmat(proj_mean,[size(domainS_proj,1) 1 ]);
    domainT_proj = domainT_proj - repmat(proj_mean,[size(domainT_proj,1) 1 ]);
    domainS_proj = L2Norm(domainS_proj);
    domainT_proj = L2Norm(domainT_proj);
    distances = EuDist2(domainT_proj,domainS_proj);
    %% class means of distances
    classMeanDist = zeros(size(distances,1),num_class);
    for i = 1:num_class
        classMeanDist(:,i) = mean(distances(:,domainS_labels==i),2);
    end
    expMatrix = exp(-classMeanDist);
    probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
    [prob,predLabels] = max(probMatrix');
    pseudoLabels = predLabels;
    W = constructW1([domainS_labels,pseudoLabels]);
    %% calculate ACC
    acc = sum(predLabels==domainT_labels)/length(domainT_labels);
    for i = 1:num_class
        acc_per_class(i) = sum((predLabels == domainT_labels).*(domainT_labels==i))/sum(domainT_labels==i);
    end
    fprintf('Iteration=%d, Acc:%0.3f,Mean acc per class: %0.3f\n', iter, acc, mean(acc_per_class));
end