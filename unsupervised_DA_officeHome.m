% =====================================================================
% Code for conference paper:
% Qian Wang, Penghui Bu, Toby Breckon, Unifying Unsupervised Domain
% Adaptation and Zero-Shot Visual Recognition, IJCNN 2019
% By Qian Wang, qian.wang173@hotmail.com
% =====================================================================
%% Loading Data:
% Features are extracted using resnet50 pretrained on ImageNet without
% fine-tuning
clear all
addpath('./utils/');
data_dir = './data';
domains = {'Art','Clipart','Product','RealWorld'};

for source_domain_index = 1:length(domains)
    load([data_dir 'OfficeHome-' domains{source_domain_index} '-resnet50-noft']);
    domainS_features = L2Norm(resnet50_features);
    domainS_labels = labels+1;
    
    for target_domain_index = 1:length(domains)
        if target_domain_index == source_domain_index
            continue;
        end
        fprintf('Source domain: %s, Target domain: %s\n',domains{source_domain_index},domains{target_domain_index});
        load([data_dir 'OfficeHome-' domains{target_domain_index} '-resnet50-noft']);
        domainT_features = L2Norm(resnet50_features);
        domainT_labels = labels+1;
        num_class = length(unique(domainT_labels));
        %% Baseline method: using 1-NN, only labelled source data for training
        fprintf('Baseline method using 1NN:\n');
        classifierType='1nn';
        acc= func_recognition(domainS_features,domainT_features,domainS_labels,domainT_labels,classifierType);
        %% Baseline method: using SVM, only labelled source data for training
        %fprintf('Baseline method using SVM:\n');
        %classifierType='svm';
        %acc= func_recognition(domainS_features,domainT_features,domainS_labels,domainT_labels,classifierType);
        %% Proposed method:
        fprintf('Proposed method using 1NN:\n');
        acc_per_class = DA_LPP(domainS_features,domainS_labels,domainT_features,domainT_labels);
        %acc_per_class = DA_LDA(domainS_features,domainS_labels,domainT_features,domainT_labels);
    end
end
