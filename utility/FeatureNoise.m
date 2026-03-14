function X_noisy = FeatureNoise(X, lambda)
if lambda == 0
    X_noisy = X;
else
    [U, W, T] = svd(X);
    rankX = nnz(W);
    k = round(lambda*rankX);  % number of noisy bases
    orthX = U(:, rankX+1:end);  % orth space
    % --- 开始修改 ---
    % 如果正交空间为空（即矩阵X是满秩的），则无法添加噪声，直接返回原数据
    if isempty(orthX) || size(orthX, 2) == 0
        X_noisy = X;
        return; % 直接退出函数
    end
    % --- 结束修改 ---

    index = randi(size(orthX, 2), 1, k);
    A = orthX(:, index);  % selected matrix
    U1 = U;
    U1(:, rankX-k+1:rankX) = A;
    X_noisy = U1*W*T;
end
end