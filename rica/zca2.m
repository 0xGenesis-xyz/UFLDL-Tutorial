function [Z, V] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%
avg = mean(x, 1);
x0 = bsxfun(@minus, x, avg);
sigma = x0*x0'/size(x0, 2);
[U,S,V] = svd(sigma);
xPCAWhite = diag(1./sqrt(diag(S)+epsilon))*U'*x0;
Z = U*xPCAWhite;
