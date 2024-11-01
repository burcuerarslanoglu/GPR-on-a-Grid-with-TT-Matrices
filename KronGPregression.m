function [Mu, Sigma, timing] = KronGPregression(X, Xtrain, y, Xtest, hyp)
    D =  length(X); % X = {X1 X2 ... XD}
    N = length(y);  % Number of training points
    M =  size(Xtest,1); % X_test : {M x D}
    
    sf_2 = hyp(2);
    l_2 = hyp(1);

    k = @(x1, x2) sf_2 * exp(-sum((x1 - x2).^2) / (2 * l_2));
    
    % Timing each section
    timing = struct();

    K_d = cell(D, 1);
    K_inv = cell(D, 1);
    K_cross_d = cell(D, 1);
    K_test_d = cell(D, 1);
    
    tic;
    for d = 1:D
        Nd = length(X{d});
        K_d{d} = zeros(Nd);
        for i = 1:Nd
            for j = i:Nd
                K_d{d}(i, j) = k(X{d}(i), X{d}(j));
                K_d{d}(j, i) = K_d{d}(i, j);  % Mirror the value
            end
        end
        K_inv{d} = inv(K_d{d}+ 1e-2 *eye(size(K_d{d},1)));     
    end
    timing.K_train = toc;

    tic;
    % Covariance Matrix K(X_*, X) for Cross Terms
    K_cross = zeros(M, N);
    for i = 1:M
        for j = 1:N
            K_cross(i, j) = k(Xtest(i, :), Xtrain(j, :));
        end
    end
    timing.K_cross = toc;
  
    tic;
    % Covariance Matrix K(X_*, X_*) for Test Points
    coordX_test = cell(D, 1);
    for d = 1:D
        coordX_test{d} = unique(Xtest(:, d)); % Extract unique coordinates along dimension d
        Md = length(coordX_test{d});
        K_test_d{d} = zeros(Md, Md);
        for i = 1:Md
            for j = i:Md
                K_test_d{d}(i, j) = k(coordX_test{d}(i), coordX_test{d}(j));
                K_test_d{d}(j, i) = K_test_d{d}(i, j); 
            end
        end
    end
    K_test = K_test_d{D};
    for d = D-1:-1:1
        K_test = kron(K_test, K_test_d{d});
    end
    timing.K_test = toc;

    tic;
    % Compute alpha = (KD^-1 ⊗ ... ⊗ K1^-1) \ y 
    alpha = kron_mvprod(K_inv, y);
    % A = kron_mvprod(K_inv, K_cross');
    % A = reshape(A, size(K_cross, 2), []);
    A = apply_kron_mvprod_to_matrix(K_inv, K_cross');
    timing.InverseK = toc;

    tic;
    % Compute the predictive mean and covariance Sigma
    Mu = K_cross * alpha;
    Sigma = K_test - K_cross * A;
    timing.Multiplications = toc;

end
