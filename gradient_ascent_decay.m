function [weights,error_per_iter] = gradient_ascent_decay(Xtrain,Ytrain,initial_step_size,iterations)

    % Function to perform gradient descent with a decaying step size for
    % logistic regression.
    % Usage: [weights,error_per_iter] = gradient_descent(Xtrain,Ytrain,step_size,iterations)
    
    % The parameters to this function are exactly the same as the
    % parameters to gradient descent with fixed step size.
    
    % initial_step_size : This parameter refers to the initial value of the step
    % size. The actual step size to update the weights will be a value
    % that is (initial_step_size * some function that decays over time)
    % some good choices for this function might by 1/n or 1/sqrt(n).
    % Experiment with such functions, and initial step size until you get
    % good performance.
%     NewCol = ones(size(Xtrain,1),1);
%     %Add new column
%     Xtrain = [NewCol Xtrain];
    
    weights = ones(size(Xtrain,2),1); % P x 1 vector of initial weights
    error_per_iter = zeros(iterations,1); % error_per_iter(i) records training error in iteration i of GD.
    % dont forget to update these values within the loop!
    
    % FILL IN THE REST OF THE CODE %
    
    Y_unchanged = Ytrain;
    
    N_train = size(Xtrain, 1);
    P_features = size(Xtrain, 2);

    Ytrain = double(Ytrain);
    Ytrain(Ytrain==0) = -1;
    Y_replicate = repmat(transpose(Ytrain),P_features,1);

    %Y_train is given as 0's and 1's lets convert it to -1's and 1's
     for iter = [1:iterations]
        a = exp(-1.*transpose(Ytrain).*(transpose(weights)*transpose(Xtrain)))./...
            (ones(1,N_train)+exp(-1*transpose(Ytrain).*(transpose(weights)*transpose(Xtrain))));
        weights = weights+(.0015/(sqrt(iter)))*((Y_replicate.*transpose(Xtrain))*transpose(a));
        
        exponent = transpose(weights)* Xtrain'; 
        p_y = 1 ./ ( 1 + exp(-exponent'));
        Y_hat = p_y > 0.5;
        error_per_iter(iter) = sum(Y_unchanged ~= Y_hat) / N_train;
     end


end

