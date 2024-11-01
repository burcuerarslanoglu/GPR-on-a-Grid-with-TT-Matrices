function [Mu, Sigma, Kernel, timing] = GPregression(Xtrain, y, Xtest, hyp)
    [N, D] =  size(Xtrain);
    M =  size(Xtest,1);
    
    sf_2 = hyp(2);
    l_2 = hyp(1);

    k = @(x1, x2) sf_2 * exp(-sum((x1 - x2).^2) / (2 * l_2));
    
    % Timing each section
    timing = struct();

    tic;
    % Covariance Matrix K(X, X) for Training Points
    K_train = zeros(N, N);
    for i = 1:N
        for j = 1:N
            K_train(i, j) = k(Xtrain(i, :), Xtrain(j, :));
        end
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
    K_test = zeros(M, M);
    for i = 1:M
        for j = 1:M
            K_test(i, j) = k(Xtest(i, :), Xtest(j, :));
        end
    end
    timing.K_test = toc;
    
    Kernel = K_train;

    tic;
    K_train = K_train + 1e-1 * eye(N);
    inv_K_train = inv(K_train);
    timing.InverseK = toc;
    
    tic;
    Mu = K_cross * inv_K_train * y;
    Sigma = K_test - K_cross * inv_K_train * K_cross';
    timing.Multiplications = toc;

end