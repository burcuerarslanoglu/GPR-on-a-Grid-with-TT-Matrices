function x = kron_mvprod(K_matrices, b)
    % K_matrices is a cell array of matrices (K1, K2, K3, etc.)
    % b is the vector we are multiplying
    D = numel(K_matrices);
    x = b;
    for d = D:-1:1
        Gd = size(K_matrices{d}, 2);
        x = reshape(x, Gd, []);
        x = K_matrices{d} * x;
        x = x';
    end
    x = x(:);  % Flatten the result
end
