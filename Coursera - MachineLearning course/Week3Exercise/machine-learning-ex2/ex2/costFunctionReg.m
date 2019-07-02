function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% theta (3*1), X(100*3), y(100*1)
%            , X(i,:) = (1*3)
step_iR = 0;
step_jThe = 0;
%size(theta);
%size(X);
%size(y);
for i = 1:m;
    _Yi = -y(i)*log(sigmoid((theta')*X(i,:)'));
    _1_Yi = (1-y(i))*log(1-sigmoid((theta')*X(i,:)'));
    step_iR += _Yi - _1_Yi;
    end
for i = 1:length(size(theta,1));
    step_jThe += theta(i,1).^2; 
    end
J = (1/m)*step_iR + (lambda/(2*m))*step_jThe;

for j = 1:size(X,2);
    gradJ_sum = 0;
    regValue = 0;
    for i = 1:m;
        gradJ_sum += (sigmoid((theta')*X(i,:)')-y(i))*X(i,j);       
        end
    if j > 1
        regValue = (lambda*theta(j,1))/m;
        endif 
    grad(j) = (1/m)*(gradJ_sum) + regValue;
    end
% =============================================================

end
