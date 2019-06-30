function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%
%new_X = [ones(length(y), 1) X];
new_X = X;
inv_X = new_X' * new_X;
inv_X = pinv(inv_X);
theta = inv_X*new_X';
theta *= y;



% ---------------------- Sample Solution ----------------------




% -------------------------------------------------------------


% ============================================================

end
