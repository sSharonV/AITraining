function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


temp_alpha = 1/m;
	new_tetha = zeros(length(theta),1);
	
	for j = 1:length(theta)
		toSum = 0;	
		for i = 1:m
			prediction = theta' * (X(i,:))';
			toSum += (prediction - y(i))*(X(i,j));
			end
		new_tetha(j) = theta(j) - alpha*temp_alpha*toSum;
		end
	theta = new_tetha;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
