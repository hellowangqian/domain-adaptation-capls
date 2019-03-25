function W = constructW2(feat1,feat2,options)
k = options.k;
len1 = size(feat1,1);
len2 = size(feat2,1);
W = zeros(len1+len2);
dist21 = EuDist2(feat2,feat1);
%dist22 = EuDist2(feat2,feat2);
[~,sortIndex] = sort(dist21,2);

for i = 1:len2
    W(len1+i,sortIndex(i,1:k)) = 1;
    W(sortIndex(i,1:k),len1+i) = 1;
end
W(len1+1:end,len1+1:end) = constructW(feat2,options);
