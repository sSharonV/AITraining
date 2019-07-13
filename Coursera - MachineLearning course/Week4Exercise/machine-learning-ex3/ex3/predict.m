function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%adds first column of '1' to X
X = [ones(m,1) X];

for i=1:m;
    %  a1 is column-vector of the original input (TrainingExample = i)
    a1 = X(i,:)';
    % we need to compute column-vector from theta(i,0)*x0 + theta(i,1)*x1...
    % for 1=<i<=25
    z2 = Theta1*a1;
    a2 = [1; sigmoid(z2)];

    % to calculate a3, we need z3 first
    z3 = Theta2*a2;
    a3 = sigmoid(z3);

    % find max in column-vector
    [a, b] = max(a3, [], 1);
    p(i) = b;
    end





% =========================================================================


end
