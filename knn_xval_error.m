function [error] = knn_xval_error(X, Y, K, part, distFunc)
% KNN_XVAL_ERROR - KNN cross-validation error.
%
% Usage:
%
%   ERROR = knn_xval_error(X, Y, K, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the K-NN algorithm on the 
% given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used.
%
% Note that N = max(PART), corresponding to the number of folds.
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, K_NEAREST_NEIGHBOURS

% FILL IN YOUR CODE HERE

nfolds = max(part);
options = nchoosek(nfolds,nfolds-1);

pctError = zeros(options,1);

%k_nearest_neighbours(Xtrain,Ytrain,Xtest,K,distfunc)
for i=1:options
    exclude = mod((i+nfolds - 2),nfolds)+1;
    Xtrain = X(part~=exclude,:);
    Ytrain = Y(part~=exclude,:);
    Xtest = X(part == exclude,:);
    Yactual = Y(part == exclude,:);    
    Yhat = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K,distFunc);
    totErrors = sum(Yactual ~= Yhat);
    pctError(i) = totErrors / length(Yhat);    
end

error = mean(pctError);