function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%   for-loop solutions over the examples and the desired clusters

for i = 1:size(X,1) % Loop over all examples
    d_powered = zeros(K, 1);    % matrix or N-dimensional array whose elements are all equal to the IEEE representation for positive infinity. (1/0, e^800)
    for desired_clus = 1:K
        % vectorized solution to calc the sum of squared error
        diff = X(i,:)' - centroids(desired_clus,:)';   % calcs the means of chosen example accords to other centorids (transpose must be used)
        d_powered = diff' * diff;           % calcs the squared root of the calculated means
        d_findMin(desired_clus) = d_powered;
        end
    [value, idx(i)] = min(d_findMin);
    end
% =============================================================
end

