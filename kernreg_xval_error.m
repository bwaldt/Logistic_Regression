function [error] = kernreg_xval_error(X, Y, sigma, part)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(X, Y, SIGMA, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the kernel regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used.
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KERNEL_REGRESSION

% FILL IN YOUR CODE HERE

nfolds = max(part);
options = nchoosek(nfolds,nfolds-1);

pctError = zeros(options,1);

%kernel_regression(Xtrain,Ytrain,Xtest,sigma)
for i=1:options
    exclude = mod((i+nfolds - 2),nfolds)+1;
    Xtrain = X(part~=exclude,:);
    Ytrain = Y(part~=exclude,:);
    Xtest = X(part == exclude,:);
    Yactual = Y(part == exclude,:);    
    Yhat = kernel_regression(Xtrain,Ytrain,Xtest,sigma);
    totErrors = sum(Yactual ~= Yhat);
    pctError(i) = totErrors / length(Yhat);
end
error = mean(pctError);
    
