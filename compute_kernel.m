function K = compute_kernel(X1, X2, K_matrix)
    % Helper function to compute kernel sub-matrix for two sets of points.
    % Inputs:
    % - X1, X2: Sets of points in one dimension.
    % - K_matrix: Kernel matrix for this dimension.
    % Outputs:
    % - K: Computed sub-matrix for these points.

    N1 = length(X1);
    N2 = length(X2);
    K = zeros(N1, N2);
    for i = 1:N1
        for j = 1:N2
            K(i, j) = K_matrix(i, j);
        end
    end
end