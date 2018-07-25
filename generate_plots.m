% Submit your textual answers, and attach these plots in a latex file for
% this homework. 
% This script is merely for your convenience, to generate the plots for each
% experiment. Feel free to change it, as you do not need to submit this
% with your code.

% Loading the data: this loads X, and Ytrain.
load('X.mat'); % change this to X_noisy if you want to run the code on the noisy data
load('Y.mat');
load('X_noisy.mat');
% X = X_noisy;

N_folds = [2,4,8,16];
errors_xval = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the [N_folds(j)]-fold cross validation error in trial i 
errors_test = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the true test error in trial i (the entire row will be identical)


for trial = 1:100
    parts = make_xval_partition(600, 4);
    X_train = X(parts~=4,:);
    Y_train = Y(parts~=4,:);
    X_test = X(parts==4,:);
    Y_test = Y(parts==4,:);
    for i = 1:4

        part = make_xval_partition(450, N_folds(i));
        
%        errors_xval(trial,i) = kernreg_xval_error(X_train, Y_train, 1, part);
%        Yhat = kernel_regression(X_train,Y_train,X_test,1);    
%        errors_xval(trial,i) = knn_xval_error(X_train, Y_train, 1, part, 'l2');
%        Yhat = k_nearest_neighbours(X_train,Y_train,X_test,1,'l2');
          errors_xval(trial,i) = logistic_xval_error(X_train, Y_train, part);
          Yhat = logistic_regression(X_train,Y_train,X_test,.0002,500);

        testError = sum(Y_test ~= Yhat) / length(Yhat);
        errors_test(trial,i) = testError;
    end
    
end

% code to plot the error bars. change these values depending on what
% experiment you are running
y = mean(errors_xval); e = std(errors_xval); x = [2,4,8,16]; % <- computes mean across all trials
errorbar(x, y, e);
hold on;
y = mean(errors_test); e = std(errors_test); x = [2,4,8,16]; % <- computes mean across all trials
errorbar(x, y, e);
title('Original data, N = [2,4,8,16]');
xlabel('N');
xlim([0 20])
ylabel('Error');
legend('N-Fold Error','Test Error');
hold off;