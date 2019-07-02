function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% g is the size of Z, so their are some dimensions to be considered,
% but with the element-wise operations: './' and '.^' the problem can be considered done.

g = 1./(1+(e.^(-z)));

% =============================================================
end
