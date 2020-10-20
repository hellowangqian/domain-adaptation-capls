%% Cross Domain Zero-Shot Classification
% Author: Qian Wang
% Date  : 18 Aug 2018
% Update: 10 Nov 2018
% Update: 08 Jan 2019
%% Loading Data:
% Features are extracted using resnet50 pretrained on ImageNet without
% fine-tuning
%clear all
addpath('./utils/');
%data_dir = '/mnt/HD2T/DomainAdaptation/OfficeHomeDataset_10072016/';
data_dir = 'E:\DomainAdaptation\OfficeHomeDataset_10072016\';
domainSet = {'Art','Clipart','Product','RealWorld'};
num_trial=5;
for source_domain_index = 1:length(domainSet)
    load([data_dir 'OfficeHome-' domainSet{source_domain_index} '-resnet50-noft']);
    %load([data_dir 'officeHome-cvae-rw2ar.mat']);
    %ytrain = ytrain+1;
    sourceDomain_features = L2Norm(resnet50_features);
    sourceDomain_labels = labels+1;
    
    for target_domain_index = 1:length(domainSet)
        if target_domain_index == source_domain_index
            continue;
        end
        for split_trial = 1:num_trial
            fprintf('\n%s->%s, split_trial = %d\n',domainSet{source_domain_index},domainSet{target_domain_index}, split_trial);
            load([data_dir 'OfficeHome-' domainSet{target_domain_index} '-resnet50-noft']);
            targetDomain_features = L2Norm(resnet50_features);
            targetDomain_labels = labels+1;
            num_class = 65;
            load([data_dir 'instanceSplit_officehome_unseen30_20200410.mat']);
            test_class = targetDomain_unseenClass{split_trial}{target_domain_index};
            train_class = ~test_class;
            
            %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Baseline 0 (sourceOnly): 1-NN using all classes from sourceDomain
            %             as training data
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            trainFeatures = sourceDomain_features;
            trainFeatures = L2Norm(trainFeatures);
            testFeatures = targetDomain_features(targetDomain_splitFlag{split_trial}{target_domain_index}==2,:);
            testFeatures = L2Norm(testFeatures);
            trainLabels = sourceDomain_labels;
            testLabels = targetDomain_labels(targetDomain_splitFlag{split_trial}{target_domain_index}==2);
            fprintf('\n Training on A, test on B, using 1NN: ');
            classifierType = '1nn';
            acc= func_recognition(trainFeatures,testFeatures,trainLabels,testLabels,classifierType);
            results.sourceOnly.acc_per_class(source_domain_index,target_domain_index,split_trial,:) = acc;
            results.sourceOnly.acc_seen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(train_class)));
            results.sourceOnly.acc_unseen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(1-train_class)));  
            %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Baseline 1: 1-NN using all classes from sourceDomain
            %             and known classes from targetDomain as training data
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            trainFeatures = [sourceDomain_features; targetDomain_features(targetDomain_splitFlag{split_trial}{target_domain_index}==1,:)];
            trainFeatures = L2Norm(trainFeatures);
            testFeatures = targetDomain_features(targetDomain_splitFlag{split_trial}{target_domain_index}==2,:);
            testFeatures = L2Norm(testFeatures);
            trainLabels = [sourceDomain_labels, targetDomain_labels(targetDomain_splitFlag{split_trial}{target_domain_index}==1)];
            testLabels = targetDomain_labels(targetDomain_splitFlag{split_trial}{target_domain_index}==2);
            
            fprintf('\n Training on A + known B, test on B, using 1NN: ');
            classifierType = '1nn';
            acc= func_recognition(trainFeatures,testFeatures,trainLabels,testLabels,classifierType);
            results.baseline1nn.acc_per_class(source_domain_index,target_domain_index,split_trial,:) = acc;
            results.baseline1nn.acc_seen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(train_class)));
            results.baseline1nn.acc_unseen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(1-train_class)));  
            % Baseline 2 SVM
            %         fprintf('\n Training on A + known B, test on B, using SVM: ');
            %         classifierType = 'svm';
            %         acc = func_recognition(trainFeatures,testFeatures,trainLabels,testLabels,classifierType);
            %         acc_per_class_baseline2{split_trial}{target_domain_index} = acc;
            %         acc_known_per_class_baseline2{split_trial}{target_domain_index} = mean(acc(logical(train_class)));
            %         acc_unseen_per_class_baseline2{split_trial}{target_domain_index} = mean(acc(logical(1-train_class)));
            %         fprintf('Acc known:%f,Acc unseen:%f\n',acc_known_per_class_baseline2{split_trial}{target_domain_index},acc_unseen_per_class_baseline2{split_trial}{target_domain_index});
            %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Baseline 3: PCA/LDA/LPP on training data from both domains
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Baseline 3.1
            clear options;
            options.ReducedDim = 1024;
            options.NeighbourMode = 'Supervised';
            options.k = 150;
            options.WeightMode = 'Binary';
            options.gnd = trainLabels';
            options.alpha = 1;
            %[P,~] = PCA(trainFeatures,options);
            %[P,~] = LDA(trainLabels',options,double(trainFeatures));
            W = constructW1(trainLabels);
            P = LPP(trainFeatures,W,options);
            
            trainFeatures_proj = trainFeatures*P;
            testFeatures_proj = testFeatures*P;
            meanTrainFeatures = mean(trainFeatures_proj);
            trainFeatures_proj = trainFeatures_proj-repmat(meanTrainFeatures,[size(trainFeatures,1) 1]);
            testFeatures_proj = testFeatures_proj-repmat(meanTrainFeatures,[size(testFeatures,1) 1]);
            trainFeatures_proj = L2Norm(trainFeatures_proj);
            testFeatures_proj = L2Norm(testFeatures_proj);
            prototypes = zeros(num_class,options.ReducedDim);
            for i = 1:num_class
                prototypes(i,:) = mean(trainFeatures_proj(trainLabels==i,:));
            end
            prototypes = L2Norm(prototypes);
            fprintf('\n Training on A + known B, test on B, using Diemsnionality Reduction: ');
            classifierType = '1nn';
            acc= func_recognition(prototypes,testFeatures_proj,[1:65],testLabels,classifierType);
            results.lpp1nn.acc_per_class(source_domain_index,target_domain_index,split_trial,:) = acc;
            results.lpp1nn.acc_seen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(train_class)));
            results.lpp1nn.acc_unseen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(1-train_class)));  
            
            % 3.2 using a different recognition method
            fprintf('\n Training on A + known B, test on B, using Diemsnionality Reduction: ');
            classifierType = 'nc';
            acc= func_recognition(trainFeatures_proj,testFeatures_proj,trainLabels,testLabels,classifierType);
            results.lppnc.acc_per_class(source_domain_index,target_domain_index,split_trial,:) = acc;
            results.lppnc.acc_seen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(train_class)));
            results.lppnc.acc_unseen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(1-train_class)));  
            
            %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Baseline 4: BiDiLEL
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % calculate the class-level representations from regular image
            % representation
            train_class1 = train_class;
            trainFeatures_A = double(sourceDomain_features);
            trainLabels_A = double(sourceDomain_labels);
            trainFeatures = double(targetDomain_features(targetDomain_splitFlag{split_trial}{target_domain_index}==1,:));
            trainLabels = double(targetDomain_labels(targetDomain_splitFlag{split_trial}{target_domain_index}==1));
            testFeatures = double(testFeatures);
            train_class = zeros(1,num_class);
            for i = 1:num_class
                A_prototypes(i,:) = mean(trainFeatures_A(trainLabels_A==i,:));
            end
            A_prototypes = L2Norm(A_prototypes);
            
            % bottom up embedding
            clear options;
            options.ReducedDim = 1024 ;
            options.NeighbourMode = 'supervised';
            options.k = 200;
            options.WeightMode = 'binary';
            options.gnd = trainLabels';
            options.alpha = 1;
            W = constructW1(trainLabels);
            P = LPP(trainFeatures,W,options);
            %[P,~] = PCA(trainFeatures,options);
            %[P,~] = LDA(trainLabels',options,double(trainFeatures));
            
            trainFeatures_proj = double(trainFeatures*P);
            testFeatures_proj = double(testFeatures*P);
            %trainFeatures_proj = double(trainFeatures);
            %testFeatures_proj = double(testFeatures);
            meanTrainFeatures = mean(trainFeatures_proj);
            trainFeatures_proj = trainFeatures_proj-repmat(meanTrainFeatures,[size(trainFeatures,1) 1]);
            testFeatures_proj = testFeatures_proj-repmat(meanTrainFeatures,[size(testFeatures,1) 1]);
            dim_y = size(trainFeatures_proj,2);
            
            % top down embedding
            B_prototypes = zeros(num_class,dim_y);
            for i = 1:num_class
                if sum(trainLabels==i)>0
                    train_class(i) = 1;
                    B_prototypes(i,:) = mean(trainFeatures_proj(trainLabels==i,:));
                end
            end
            test_class = 1-train_class;
            train_class = logical(train_class);
            test_class = logical(test_class);
            
            B_prototypes_known = B_prototypes(train_class,:);
            B_prototypes_known = L2Norm(B_prototypes_known);
            B_prototypes_unseen = zeros(sum(test_class),size(trainFeatures_proj,2));
            % SVR
            cmd = ['-s 3 -t 0 -c 1 -h 0 -q'];
            factor = 1;
            for j=1:dim_y
                model = svmtrain(B_prototypes_known(:,j)*factor,A_prototypes(train_class,:),cmd);
                [B_prototypes_unseen(:,j), accuracy, prob_estimates]=svmpredict(zeros(sum(test_class),1),A_prototypes(test_class,:),model, '-q');
            end
            B_prototypes_unseen = L2Norm(B_prototypes_unseen/factor);

            B_prototypes(train_class,:) = B_prototypes_known;
            B_prototypes(test_class,:) = B_prototypes_unseen;
            
            distances = EuDist2(testFeatures_proj,B_prototypes);
            [~,preds] = min(distances');
            % calculate ACC
            acc_per_image = sum(preds==testLabels)/length(testLabels);
            for i = 1:num_class
                acc(i) = sum((preds == testLabels).*(testLabels==i))/sum(testLabels==i);
            end
            train_class = train_class1;
            fprintf('\n Training on A + known B, test on B, using BiDiLEL: Acc:%f,Mean acc per class: %f\n', mean(acc_per_image), mean(acc));
            results.bidilel.acc_per_class(source_domain_index,target_domain_index,split_trial,:) = acc;
            results.bidilel.acc_seen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(train_class)));
            results.bidilel.acc_unseen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(1-train_class)));  
            %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Baseline 5: Direct Mapping Learning
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            trainFeaturesA = sourceDomain_features;
            trainFeaturesA = L2Norm(double(trainFeaturesA));
            trainFeaturesB = targetDomain_features(targetDomain_splitFlag{split_trial}{target_domain_index}==1,:);
            trainFeaturesB = L2Norm(double(trainFeaturesB));
            trainFeatures = [sourceDomain_features; targetDomain_features(targetDomain_splitFlag{split_trial}{target_domain_index}==1,:)];
            trainFeatures = L2Norm(trainFeatures);
            testFeatures = targetDomain_features(targetDomain_splitFlag{split_trial}{target_domain_index}==2,:);
            testFeatures = L2Norm(double(testFeatures));
            trainLabelsA = double(sourceDomain_labels);
            trainLabelsB = double(targetDomain_labels(targetDomain_splitFlag{split_trial}{target_domain_index}==1));
            trainLabels = [sourceDomain_labels, targetDomain_labels(targetDomain_splitFlag{split_trial}{target_domain_index}==1)];
            testLabels = targetDomain_labels(targetDomain_splitFlag{split_trial}{target_domain_index}==2);
            numTrainA = length(trainLabelsA);
            numTrainB = sum(targetDomain_splitFlag{split_trial}{target_domain_index}==1);
            classMeanA = [];
            for i = 1:num_class
                classMeanA(i,:) = mean(trainFeaturesA(trainLabelsA == i, :));
            end
            classMeanA = L2Norm(classMeanA);
            Y = [];
            for i = 1:length(trainLabelsB)
                Y(i,:) = classMeanA(trainLabelsB(i),:);
            end
            Wb = constructW1(trainLabelsB);
            Db = diag(sum(Wb,1));
            Lb = Db-Wb;
            X = double(trainFeaturesB);
            Slb = X'*Lb*X;
            Slb = (Slb+Slb')/2;
            % Using closed-form solution
            P = (X'*X+0.01*eye(size(X,2))+0.01*Slb)\X'*Y;
            testFeatures_proj = testFeatures*P;
            trainFeaturesB_proj = trainFeaturesB*P;
            trainFeaturesB_proj = L2Norm(trainFeaturesB_proj);
            trainFeatures_proj = [trainFeaturesA;trainFeaturesB_proj];
            classMean = [];
            for i = 1:length(trainLabels)
                classMean(i,:) = mean(trainFeatures(trainLabels == i, :));
            end
            
            testFeatures_proj = L2Norm(testFeatures_proj);
            % Using SVR
            %         cmd = ['-s 3 -t 0 -c 1 -h 0 -q'];
            %         factor = 1;
            %         for j=1:size(Y,2)
            %             model = svmtrain(Y(:,j)*factor,X,cmd);
            %             [testFeatures_proj(:,j), accuracy, prob_estimates]=svmpredict(zeros(length(testLabels),1),testFeatures,model, '-q');
            %         end
            % recognition
            fprintf('\n Training on A + known B, test on B, using Direct Mapping: ');
            classifierType = '1nn';
            acc= func_recognition(classMean,testFeatures_proj,[1:65],testLabels,classifierType);
            results.dmapping.acc_per_class(source_domain_index,target_domain_index,split_trial,:) = acc;
            results.dmapping.acc_seen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(train_class)));
            results.dmapping.acc_unseen(source_domain_index,target_domain_index,split_trial) = mean(acc(logical(1-train_class)));  
            
        end
    end
end
save([data_dir 'Baseline_results_unseen30_20200415.mat'], 'results');
%% Print out the results in latex format
data_dir = 'E:\DomainAdaptation\OfficeHomeDataset_10072016\';
load([data_dir 'Baseline_results_unseen30_20200415.mat']);
avg = 0;
for sourceIndex = 3%1:4
    for targetIndex = 1:4
        if sourceIndex == targetIndex
            continue;
        end
        acc_seen = zeros(1,5);
        acc_unseen = zeros(1,5);
        h = zeros(1,5);
        for trialIndex = 1:5
            acc_seen(trialIndex) = results.bidilel.acc_seen(sourceIndex,targetIndex,trialIndex);
            acc_unseen(trialIndex) = results.bidilel.acc_unseen(sourceIndex,targetIndex,trialIndex);
        end
        h = 2*acc_seen.*acc_unseen./(acc_seen+acc_unseen);
        avg = avg + mean(h);
        fprintf('&$%2.1f\\pm%2.1f$&$%2.1f\\pm%2.1f$&$%2.1f\\pm%2.1f$',mean(acc_seen)*100,std(acc_seen)/sqrt(5)*100,mean(acc_unseen)*100,std(acc_unseen)/sqrt(5)*100,mean(h)*100,std(h)/sqrt(5)*100);
        %fprintf('%2.1f, %2.1f, %2.1f\n',mean(acc_seen)*100,mean(acc_unseen)*100,mean(h)*100);
    end
end
fprintf('\\\\\n');
%avg = avg/12;
%fprintf('&%2.1f\\\\',avg*100);

