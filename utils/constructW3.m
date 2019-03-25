function W = constructW3(domainS_proj,domainT_proj,domainS_labels,pseudoLabels,options);
len1 = size(domainS_proj,1);
len2 = size(domainT_proj,1);
W = zeros(len1+len2);
W(1:len1,1:len1) = constructW1(domainS_labels);
num_class = max(domainS_labels(:));
for i = 1:num_class
    W(1:len1,len1+1:end) = W(1:len1,len1+1:end) + 2*double(domainS_labels==i)'*double(pseudoLabels==i);
end
W(len1+1:end,len1+1:end) = 2*constructW1(pseudoLabels);
W = max(W,W');