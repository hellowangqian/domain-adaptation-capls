function P = DLPP(data,classMean,W,B,options)
data = double(data);
W = double(W);
D = diag(sum(W,2));
L = D - W;

B = double(B);
E = diag(sum(B,2));
H = E-B;

Sl = data'*L*data;
Sh = classMean'*H*classMean;
Sl = (Sl+Sl')/2;
Sh = (Sh+Sh')/2;

Sl = Sl + options.alpha*eye(size(Sl,2));

[P,Diag] = eigs(double(Sh),double(Sl),options.ReducedDim,'la',options);
%[P,Diag] = eigs(double(Sh)-0.00001*double(Sl),options.ReducedDim,'la',options);

for i = 1:size(P,2)
    if (P(1,i)<0)
        P(:,i) = P(:,i)*-1;
    end
end