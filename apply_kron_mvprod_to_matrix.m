function result = apply_kron_mvprod_to_matrix(K_matrices, B)
    total_rows = 1;
    for d = 1:numel(K_matrices)
        total_rows = total_rows * size(K_matrices{d}, 1);
    end
    
    [~, num_columns] = size(B);
    result = zeros(total_rows, num_columns);
    
    for i = 1:num_columns
        result(:, i) = kron_mvprod(K_matrices, B(:, i));
    end
end
