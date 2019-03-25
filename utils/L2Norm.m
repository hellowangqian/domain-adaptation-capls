function y = L2Norm(x)
% x is a feature matrix: one example in a row
% 
y = x./repmat(sqrt(sum(x.^2,2)),[1 size(x,2)]);