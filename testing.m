clear
load('X.mat')
load('Y.mat')
parts = make_xval_partition(600, 4);
X_train = X(parts~=4,:);
Y_train = Y(parts~=4,:);
X_test = X(parts==4,:);
Y_test = Y(parts==4,:);

% test_vector = linspace(.0001,.00025,1000);
% err = linspace(.001,.005,1000);


b = glmfit(X_train,Y_train,'binomial');
yfit = glmval(b,X_test,'logit');
yfit2 = logistic_regression(X_train,Y_train,X_test,.015,500);
yfit = yfit > .5;
err1 = sum(yfit ~= yfit2);
res(:,1) = yfit;
res(:,2) = yfit2;
b
err1



% plot(test_vector,err)

