function labels = kernel_regression(Xtrain,Ytrain,Xtest,sigma)

    % Function that implements kernel regression on the given data (binary classification)
    % Usage: labels = kernel_regression(Xtrain,Ytrain,Xtest)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % sigma : width of the (gaussian) kernel.
    % labels : return an M x 1 vector of predicted labels for testing data.
    
        %Given an unknown point we find point closest
        
    Yt(Ytrain == 0) = -1;
    Yt(Ytrain == 1) = 1;
    Yt = Yt';
    
    [N,P] = size(Xtest);
    labels = zeros(1,N); %Initialize labels
    
    %Itterate through every test point
    for i=1:N
        distances  = my_norm('l2',Xtrain,Xtest(i,:));
        kernMat = exp(-distances.^2/sigma.^2); % k(x,xi)
        val = kernMat*Yt; % sum of kernDistance * Label for all training points (?ni=1K(x,xi)yi)
        labels(i) = val > 0.0 ;        
    end
    
    labels = labels';

end



function dist_vector = my_norm(distfunc,train,test_point)
    %Function calculates the distance between Training set and a test point
    
    %distfunc = 'l1','l2','linf'
    %train = Large matrix of training points. Dimensions NxP
    %test poin = Dimensions 1xP
    %dist_vector = 1xN vector of distances between test point and every
    %other point
    
    %CODE
    [N,P] = size(distfunc);
    train_transpose = transpose(train); %Dimensions = PxN
    test_point_transpose = repmat(transpose(test_point),1,N);
    diff = train_transpose - test_point_transpose;
    switch distfunc
        case 'l1'
            p = 1;
        case 'l2'
            p = 2;
        case 'linf'
            p = Inf;
        otherwise
            error('Invalid distfunc');   
    end
    
    if p ~= Inf
        dist_vector = sum(abs(diff).^p,1).^(1/p);
    else
        dist_vector = max(abs(diff),[],1);
    end
end
    
    