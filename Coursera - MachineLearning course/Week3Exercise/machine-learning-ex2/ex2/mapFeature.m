function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%   * every given value need to be within 'max degree' of equ. (like: x1*x2^5, and not x1^4*x2^3)
%   max degree = sum of powers in poly. equ.
%   Inputs X1, X2 must be the same size
%   example for alg. :
%   i,j
%   1,0 -> x1^1 * x2^0 = x1
%   1,1 -> x1^0 * x2^1 = x2

%   2,0 -> x1^2 * x2^0 = x1^2
%   2,1 -> x1^1 * x2^1 = x1*x2
%   2,2 -> x1^0 * x2^2 = x2^2

%   3,0 -> x1^3 * x2^0 = x1^3
%   3,1 -> x1^2 * x2^1 = x1^2 * x2
%   3,2 -> x1^1 * x2^2 = x1 * x2^2
%   3,3 -> x1^0 * x2^3 = x2^3
%
%   out(:, end + 1) = ... -> appends new column to 'out'
degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end