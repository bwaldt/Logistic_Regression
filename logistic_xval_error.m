function [error] = logistic_xval_error(X, Y, part)
% LOGISTIC_XVAL_ERROR - Logistic regression cross-validation error.
%
% Usage:
%
%   ERROR = logistic_xval_error(X, Y, PART)
%
% Returns the average N-fold cross validation error of the logistic regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, LOGISTIC_REGRESSION


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
    Yhat = logistic_regression(Xtrain,Ytrain,Xtest,.0002,500);
    totErrors = sum(Yactual ~= Yhat);
    pctError(i) = totErrors / length(Yhat);
end
error = mean(pctError);