function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE

%Assuming n is a scalar and n_folds is a scalar too?

%Attempt 1 = Classes are unequal
%part = randi(n_folds,1,n);
rem = mod(n,n_folds); %remainder
quo = (n-rem)/(n_folds); %quotient
part = ceil([1:(quo*n_folds)]./quo); %Generates repeating numbers
%we only have quotient*n_folds number of elements, fill in remainder
part = horzcat(part,1:1:rem); %Concatenates repeating numbers and remainders
%Now that we have the numbers, lets permutate
part = part(randperm(n));
end
