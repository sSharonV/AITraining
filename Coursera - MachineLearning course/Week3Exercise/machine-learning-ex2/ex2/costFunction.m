function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% theta (3*1), X(100*3), y(100*1)
%            , X(i,:) = (1*3)
step_iR = 0;
%size(theta);
%size(X);
%size(y);
for i = 1:m;
    _Yi = -y(i)*log(sigmoid((theta')*X(i,:)'));
    _1_Yi = (1-y(i))*log(1-sigmoid((theta')*X(i,:)'));
    step_iR += _Yi - _1_Yi;
    end
J = (1/m)*step_iR;

for j = 1:size(X,2);
    gradJ_sum = 0;
    for i = 1:m;
        gradJ_sum += (sigmoid((theta')*X(i,:)')-y(i))*X(i,j);       
        end
    grad(j) = (1/m)*(gradJ_sum);
    end




% =============================================================

end
