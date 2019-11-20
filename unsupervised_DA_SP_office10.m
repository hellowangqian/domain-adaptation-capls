% =====================================================================
% Code for conference paper:
% Qian Wang, Toby Breckon, Unsupervised Domain Adaptation via Structured Prediction Based Selective Pseudo-Labeling, AAAI2020
% By Qian Wang, qian.wang173@hotmail.com
% =====================================================================
%% Loading Data:
% Features are extracted using resnet101 pretrained on ImageNet without
% fine-tuning
clearvars;
addpath('./utils/');
%data_dir = './JGSA-r/data/GFKdata/';
data_dir = '../../ZSL_CrossDomain/Office10/decaf/';
data_dir = 'D:\Dropbox\Codes\ZSL_CrossDomain\Office10\decaf\';
domains = {'caltech','amazon','webcam','dslr'};
count = 0;
for source_domain_index = 1:length(domains)
    %load([data_dir domains{source_domain_index} '_zscore_SURF_L10']);
    load([data_dir domains{source_domain_index} '_decaf.mat']);
    domainS_features_ori = L2Norm(feas);
    %domainS_features_ori = feas;
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
        %my_tsne(domainS_features_ori,domainT_features,domainS_labels,domainT_labels);
        opts.ReducedDim = 128;
        X = double([domainS_features_ori;domainT_features]);
        P_pca = PCA(X,opts);
        domainS_features = domainS_features_ori*P_pca;
        domainT_features = domainT_features*P_pca;
        domainS_features = L2Norm(domainS_features);
        domainT_features = L2Norm(domainT_features);
        %my_tsne(domainS_features,domainT_features,domainS_labels,domainT_labels);
        num_class = length(unique(domainT_labels));
%         %% Baseline method: using 1-NN, only labelled source data for training
%         fprintf('Baseline method using 1NN:\n');
%         classifierType='1nn';
%         [acc_per_img,acc_per_class]= func_recognition(domainS_features,domainT_features,domainS_labels,domainT_labels,classifierType);
%         count = count+1;
%         acc1nn(count) = acc_per_img;
%         %% Baseline method: using NC, only labelled source data for training
% %         fprintf('Baseline method using NC:\n');
% %         classifierType='nc';
% %         acc= func_recognition(domainS_features,domainT_features,domainS_labels,domainT_labels,classifierType);
%         %% Baseline method: using SVM, only labelled source data for training
%         fprintf('Baseline method using SVM:\n');
%         classifierType='svm';
%         [acc_per_img, acc_per_class] = func_recognition(domainS_features,domainT_features,domainS_labels,domainT_labels,classifierType);
%         accsvm(count)=acc_per_img;
        %% Proposed method:
        using_sp = 1;
        d = 128;
        T = 11;
        if using_sp
            [acc, acc_per_class] = DA_LPP_SP(domainS_features,domainS_labels,domainT_features,domainT_labels,d,T);
        else
            [acc, acc_per_class] = DA_LPP(domainS_features,domainS_labels,domainT_features,domainT_labels,d,T);          
             %acc_per_class = DA_LDA(domainS_features,domainS_labels,domainT_features,domainT_labels);
        end
        count = count + 1;
        all_acc_per_class(count,:) = mean(acc_per_class,2);
        all_acc_per_image(count,:) = acc;
    end
end
mean_acc_per_class = mean(all_acc_per_class,1)
mean_acc_per_image = mean(all_acc_per_image,1)
save(['office10-SP-' num2str(using_sp) '-PcaDim-' num2str(opts.ReducedDim) '-LppDim-' num2str(d) '-T-' num2str(T) '.mat'],'all_acc_per_class','all_acc_per_image','mean_acc_per_class','mean_acc_per_image');
