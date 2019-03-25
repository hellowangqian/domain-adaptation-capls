% =====================================================================
% Code for conference paper:
% Qian Wang, Penghui Bu, Toby Breckon, Unifying Unsupervised Domain
% Adaptation and Zero-Shot Visual Recognition, IJCNN 2019
% By Qian Wang, qian.wang173@hotmail.com
% =====================================================================
%% Loading Data:
% Features are extracted using resnet101 pretrained on ImageNet without
% fine-tuning
clear all
addpath('./utils/');
%data_dir = './JGSA-r/data/GFKdata/';
data_dir = '../Office10/decaf/';
domains = {'caltech','amazon','dslr','webcam'};

for source_domain_index = 2%1:length(domains)
    %load([data_dir domains{source_domain_index} '_zscore_SURF_L10']);
    load([data_dir domains{source_domain_index} '_decaf.mat']);
    domainS_features = L2Norm(feas);
    %domainS_features = feas;
    domainS_labels = labels';
    
    for target_domain_index = 1:length(domains)
        if target_domain_index == source_domain_index
            continue;
        end
        fprintf('Source domain: %s, Target domain: %s\n',domains{source_domain_index},domains{target_domain_index});
        %load([data_dir domains{target_domain_index} '_zscore_SURF_L10']);
        load([data_dir domains{target_domain_index} '_decaf.mat']);
        domainT_features = L2Norm(feas);
        %domainT_features = feas;
        domainT_labels = labels';
        num_class = length(unique(domainT_labels));
        %% Baseline method: using 1-NN, only labelled source data for training
        fprintf('Baseline method using 1NN:\n');
        classifierType='1nn';
        acc= func_recognition(domainS_features,domainT_features,domainS_labels,domainT_labels,classifierType);
        %% Baseline method: using NC, only labelled source data for training
%         fprintf('Baseline method using NC:\n');
%         classifierType='nc';
%         acc= func_recognition(domainS_features,domainT_features,domainS_labels,domainT_labels,classifierType);
        %% Baseline method: using SVM, only labelled source data for training
%         fprintf('Baseline method using SVM:\n');
%         classifierType='svm';
%         acc= func_recognition(domainS_features,domainT_features,domainS_labels,domainT_labels,classifierType);
        %% Proposed method:
        %fprintf('Proposed method using 1NN:\n');
        acc_per_class = DA_LPP(domainS_features,domainS_labels,domainT_features,domainT_labels);
        %acc_per_class = DA_LDA(domainS_features,domainS_labels,domainT_features,domainT_labels);
    end
end
