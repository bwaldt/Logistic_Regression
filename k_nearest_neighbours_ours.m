function labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K,distfunc)

    % Function to implement the K nearest neighbours algorithm on the given
    % dataset.
    % Usage: labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % K : number of nearest neighbours used to make predictions on the test
    %     dataset. Remember to take care of corner cases.
    % distfunc: distance function to be used - l1, l2, linf.
    % labels : return an M x 1 vector of predicted labels for testing data.
    
    % YOUR CODE GOES HERE.
    
    %Given an unknown point we find loint closest
    [N,P] = size(Xtest);
    labels = zeros(1,N); %Initialize labels
    %Itterate through every test point
    for i=1:N
        distances  = my_norm(distfunc,Xtrain,Xtest(i,:));
        [sortedValues,sortedIndex] = sort(distances,'ascend');
        %We now look at the sorted Indeces and pick the majority vote of
        %the top K.
        labels(i) = mode(Ytrain(sortedIndex(1:K)));        
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