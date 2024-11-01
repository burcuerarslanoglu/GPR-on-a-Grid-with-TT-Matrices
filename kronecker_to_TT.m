function TT_cores = kronecker_to_TT(varargin)
    num_matrices = length(varargin);
    TT_cores = cell(num_matrices, 1);

    % Construction of TT-cores for each input matrix in normal order (outer product representation)
    for n = 1:num_matrices
        [I, J] = size(varargin{n});
        TT_cores{n} = reshape(varargin{n}, [1, I*J, 1]); %rank-1 connections in between 
    end
    
    % Complexity Reduction
    total_elements = prod(cellfun(@(x) numel(x), varargin)); % Full Kronecker product size
    tt_elements = sum(cellfun(@numel, TT_cores)); % Total TT elements

    fprintf('Total elements: %d\n', total_elements);
    fprintf('TT elements: %d\n', tt_elements);
end
