function B = constructB(classMean,options)
num_class = size(classMean,1);
D = EuDist2(classMean);
options.t = mean(D(:));
B = exp(-D/(2*options.t^2));
