function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% - '+1' is for bios input
% - Takes vector size 1:(25*401) and converts it to 25X401 matrix
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                %   INPUT of hidden layers, INPUT of input layer + bios 
                %   for every activator reshape all corresponding thetas
                 hidden_layer_size, (input_layer_size + 1));
% - Takes vector size (25*401 + 1):end and converts it to 10X26 matrix
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                %   INPUT of output layers, INPUT of input layer + bios 
                %   for every result reshape all corresponding thetas
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%   Adding 1 as bios to 'a1' -> #Examples X 401
a1 = [ones(m, 1) X];
%   According to Fig.2 in pdf (notice to Transpose Theta1 -> #ExamplesX401 * 401X25)
a2 = sigmoid(a1 * Theta1');
%   size(a2) = #ExamplesX25
a2 = [ones(m, 1) a2];
%   Adding 1 as bios to 'a2' -> #ExamplesX26 
%   According to Fig.2 in pdf (notice to Transpose Theta2 -> #ExamplesX26 * 26X10
a3 = sigmoid(a2 * Theta2');
h_theta = a3;

% prefer to use K instead of num_labels for clarity
K = num_labels;

#   A way for converting our output vector 'y' to an otuput vector for each output
#   with the value of 1 in the corresponding column 
correct_output = eye(num_labels)(y,:);

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#Solution with for loop

#for i = 1:m
#    for k = 1:num_labels
#        J += (-correct_output(i,k)*log(h_theta(i,k))-(1-correct_output(i,k))*log(1-h_theta(i,k)));
#        end
#    end
#J /= m;

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#Solution without for loop
#   Sum(x,2) -> returns a column vector that contains the subtotal of all items in each row added together
#   As we can see from the 'for-loop' version, we want to sum for every row its columns - so Sum(x,2)...
#   Summing over a column vector will sum all the values
J = sum(sum((-correct_output).*log(h_theta) - (1-correct_output).*log(1-h_theta), 2))/m;

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

% calculte penalty
#   Sum(x,2) -> returns a column vector that contains the subtotal of all items in each row added together
#   Again we would want to sum for every row of theta, all the columns
reg_val = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));
reg_val *= lambda/(2*m);

J += reg_val;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#Solution with for loop - according to steps in PDF
#Delta1 = zeros(size(Theta1));
#Delta2 = zeros(size(Theta2));
#for t = 1:m
    % for each outputExm., subtruct coressponding neurons of last layer -> 10X1
#    delta_3 = a3(:,t) - y_vec(:,t); 
    %26 * 1
#    delta_2 = Theta2' * delta_3 .* sigmoidGradient(a2(:,t)); 
#    D1 = D1 + delta_2(2:end) * a1(t,:);
#    D2 = D2 + delta_3 * a2(:,t)’;
#    end

#Theta1_grad = D1 / m;
#Theta1_grad = Theta1_grad(:, 2:end)
#Theta2_grad = D2 / m;
#Theta2_grad = Theta2_grad(:, 2:end)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#Solution without for loop - after recognizing the sum op. on the matrixes it's easy
#to notice that we can skip the for with matrix-multiplication
delta_3 = a3.-correct_output;
delta_2 = d2 = delta_3*Theta2(:,2:end).*sigmoidGradient(a1 * Theta1');
D1 = delta_2' * a1;
D2 = delta_3' * a2;

Theta1_grad = D1./m;
Theta2_grad = D2./m;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% calculate regularized gradient
# we can add a column of zeros to ThetaX in order to properly update j=0
reg_1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
reg_2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad += reg_1;
Theta2_grad += reg_2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
