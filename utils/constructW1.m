function W = constructW1(label)
W = zeros(length(label),length(label));
num_class = max(label(:));
for i = 1:num_class
    W = W + double(label==i)'*double(label==i);
end
