function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

%   Vectorized implementation for summerizing each feature's values
%mu = 1/m .* sum(X); %   product is 2(#features=n) X 1(vector)

%   In order to calculate the mean-Mat (mXn) we need to compute first  
%   initializing desired matrix for propabilities of each example (columns) and it's features value (rows)

%pred_per_exm = mu' * ones(1,m); %   product is nXm
%pred_per_exm = pred_per_exm';           %   we need the matrix tranposed (size(X) = mXn) so we can calculate the means of our predictions

%   Easily computing sigma2 knowing appropriate avarage value of feature

%sigma2 = 1/m .* sum((X-pred_per_exm).^2);

%   Efficient solution
mu = mean(X)';
sigma2 = var(X, 1)';

% =============================================================


end
