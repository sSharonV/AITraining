function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%   Begineer solution
%for desired_clus = 1:K
%    size_k = 0;
%    tmp_sum = zeros(n,1); % sums for each example the appropriate sum-vec of its features
%    for i = 1:m
%        if idx(i) == desired_clus
%            tmp_sum = tmp_sum + X(i,:)';
%            size_k++;
%            end
%        end
%    centroids(desired_clus, :) = (tmp_sum/size_k)';
%    end

%   Advenced solution
for desired_clus = 1:K
    % returns a matrix which contains rows of each examples which allocated to 'desired_clus'
    des_indices = find(idx==desired_clus);

    % Sum all the selected examples and calc elemnt-wise division for each feature
    centroids(desired_clus, :) = sum(X(des_indices, :)) ./ length(des_indices);
    end
% =============================================================

end

