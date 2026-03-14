function [P_final] = learn_projection_RCC_PML(X, Y_partial, HyperPara)

[n, d] = size(X); q = size(Y_partial, 2);

% ---------------- hyperparams ----------------
d_prime     = gparam(HyperPara,'d_prime',min(50,d));
alpha       = gparam(HyperPara,'alpha_initial',0.6);
decay_s     = gparam(HyperPara,'decay_s',0.98);
k           = gparam(HyperPara,'k',10);
k_graph     = gparam(HyperPara,'k_graph',10);
eta_local   = gparam(HyperPara,'eta_local',4e-4);
eta_global  = gparam(HyperPara,'eta_global',2e-4);
lambda_w    = gparam(HyperPara,'lambda_w',5e-3);    
lambda_f    = gparam(HyperPara,'lambda_f',0);       
maxIter     = gparam(HyperPara,'maxIter',50);
minMargin   = gparam(HyperPara,'minLossMargin',1e-5);
alpha_min   = gparam(HyperPara,'alpha_min',0.3);
alpha_max   = gparam(HyperPara,'alpha_max',0.8);
alpha_floor = gparam(HyperPara,'alpha_floor',0.3);

% F 更新参数
T           = gparam(HyperPara,'T',0.5);
rho_fix     = gparam(HyperPara,'rho_fix',0.9);
Kref        = gparam(HyperPara,'F_update_period',3);
etaF0       = gparam(HyperPara,'F_momentum',0.25);
etaF_min    = gparam(HyperPara,'F_momentum_min',0.10);

% W 更新参数
etaW0       = gparam(HyperPara,'W_momentum',0.6);
etaW_min    = gparam(HyperPara,'W_momentum_min',0.2);
w_min       = gparam(HyperPara,'W_min',0.05);
w_pow_mad   = gparam(HyperPara,'W_pow_mad',1.0);
w_pow_nn    = gparam(HyperPara,'W_pow_nn',1.0);
w_pow_rel   = gparam(HyperPara,'W_pow_rel',1.5);

% 排序项 (新增)
lambda_rank = gparam(HyperPara,'lambda_rank',1e-3);   % >0 开启排序保持
rank_topk   = gparam(HyperPara,'rank_topk',3);       % 0=全量；建议 3~5

% L21 行稀疏近端 (新增)
l21_tau0    = gparam(HyperPara,'l21_tau',1e-3);       % >0 开启近端
l21_tau_min = gparam(HyperPara,'l21_tau_min',5e-4);   % 退火末值（可=0）

% Early stop
patience    = gparam(HyperPara,'earlystop_patience',5);
eps_rel     = gparam(HyperPara,'earlystop_eps',1e-3);

% Anchor-Graph
n_anchor    = gparam(HyperPara,'n_anchor',min(200,max(50,round(sqrt(n)))));
k_anchor    = gparam(HyperPara,'k_anchor',3);
perplexity  = gparam(HyperPara,'perplexity',20);

% Global Graph
k_global    = gparam(HyperPara,'k_global',min(20, max(10, round(log2(n))+5)));

% Rank-aware Laplacian (保持向后兼容，非排序项)
beta_rank   = gparam(HyperPara,'beta_rank',0);
pair_samp   = gparam(HyperPara,'pair_sample',min(20000,max(0,n*(n-1)/2)));
rank_jacc   = gparam(HyperPara,'rank_jaccard_thresh',0.15);

% Correntropy 核宽度（用于 Sw）
mad_global  = median(abs(X(:) - median(X(:)))) * 1.4826;
sigma_sw    = gparam(HyperPara,'sigma_sw', max(1e-3, 2.0*mad_global));

% ---------------- init F, W, P ----------------
F = Y_partial ./ (sum(Y_partial,2) + 1e-12);  % n x q, 行归一
W = ones(d,1);                                 % d x 1, 初始全信任

try
    [coeff,~] = pca(X,'NumComponents',d_prime);
    P = coeff;
catch
    [P,~] = qr(randn(d,d_prime),0);
end
P_old = zeros(size(P));

L_local  = build_pl_graph_anchor_local(X, Y_partial, n_anchor, k_anchor, perplexity, k_graph);
L_global = build_global_graph_cosine(X, k_global);

% 初始尺度
tauZ  = robust_scale(X * P);
Xw    = X .* (sqrt(W(:)'+1e-12));
tauW  = robust_scale(Xw);

J_hist = [];

% ---------------- optimize ----------------
for it = 1:maxIter
    % (A) 统计
    mu   = mean(X,1)';         % d x 1
    sumF = sum(F,1);           % 1 x q
    mu_j = (X' * F) ./ (sumF + 1e-12);   % d x q 软类均值

    % (B) Sw 
    Sw = zeros(d,d);
    for j = 1:q
        Xc = X' - mu_j(:,j);
        Rj = sqrt(sum((X' - mu_j(:,j)).^2,1))';
        w  = exp(-(Rj.^2)/(2*sigma_sw^2)) .* F(:,j);
        Sw = Sw + (Xc * diag(w) * Xc');
    end
    Sw = Sw + 1e-6*eye(d);

    % (C) Sb
    Sb = (mu_j - mu) * diag(sumF) * (mu_j - mu)';

    % (D) 判别矩阵 B
    B = Sb;

    % —— 可选：rank-aware Laplacian（与排序项不同，保持兼容）
    if (beta_rank > 0) && (pair_samp > 0)
        Xz_tmp = X * P;
        [ng,~]  = knnsearch(Xz_tmp, Xz_tmp, 'K', min(15,n));
        ng = ng(:,2:end);
        mask_knn = sparse(repmat((1:n)',1,size(ng,2)), ng, 1, n, n);

        ii = randi(n, pair_samp, 1);
        jj = randi(n, pair_samp, 1);
        keep = (ii~=jj) & (mask_knn(sub2ind([n n],ii,jj))>0);
        ii = ii(keep); jj = jj(keep);

        if ~isempty(ii)
            Ci = Y_partial(ii,:)>0; Cj = Y_partial(jj,:)>0;
            inter_cnt = sum(Ci & Cj, 2); union_cnt = sum(Ci | Cj, 2);
            jac = zeros(size(inter_cnt)); nz = union_cnt>0;
            jac(nz) = inter_cnt(nz)./union_cnt(nz);

            fi = max(F(ii,:),[],2); fj = max(F(jj,:),[],2);
            s  = fi - fj; pij = 1./(1+exp(-s));

            jac(jac < rank_jacc) = 0;
            w_pair = jac .* abs(s) .* (pij .* (1 - pij));
            w_pos  = full(w_pair(w_pair>0));
            if ~isempty(w_pos)
                cap = percentile(w_pos, 95);
                w_pair = min(w_pair, cap);
            end
            Wp = sparse(ii, jj, w_pair, n, n); Wp = max(Wp, Wp');
            Dwp = spdiags(sum(Wp,2),0,n,n); Lrank = Dwp - Wp;

            trRank = trace((X') * Lrank * X);
            if trRank > 0, Lrank = Lrank * (n / trRank); end
            B = B + beta_rank * (X' * Lrank * X);
        end
    end

    % —— 新增：(A) 排序保持判别项
    if lambda_rank > 0
        B_rank = build_B_rank(mu_j, F, rank_topk);   % d x d
        % 能量对齐（避免过大/过小）
        trSb = trace(Sb); trBr = trace(B_rank);
        if trBr > 0 && trSb > 0
            B_rank = B_rank * (trSb / trBr);
        end
        B = B + lambda_rank * B_rank;
    end

    % (E) 混合流形
    Lloc = L_local; Lglb = L_global;
    trLloc = trace((X') * Lloc * X); if trLloc>0, Lloc = Lloc * (n / trLloc); end
    trLglb = trace((X') * Lglb * X); if trLglb>0, Lglb = Lglb * (n / trLglb); end
    Lhy = eta_local * Lloc + eta_global * Lglb;

    % (F) 特征噪声惩罚 diag(1./W)
    D_w = diag(1 ./ (W + 1e-12));

    % (G) C 矩阵
    C = (1 - alpha)*Sw + alpha*eye(d) + lambda_w * D_w + (X' * Lhy * X);
    C = (C + C')/2 + 1e-7*eye(d);

    if it == 1 || mod(it,10)==0
        sC = svd(C); condC = sC(1)/max(sC(end),1e-12);
        fprintf('      [diag] cond(C)=%.3e, trace(Sw)=%.3e, trace(Sb)=%.3e\n', condC, trace(Sw), trace(Sb));
    end

    % (H) GEVD
    [V,Dv] = eig(B, C);
    ev = real(diag(Dv)); [~,ord] = sort(ev,'descend');
    P = real(V(:, ord(1:d_prime)));

    if l21_tau0 > 0 || l21_tau_min > 0
        t = it / maxIter;
        tau_now = l21_tau0 * (1 - t) + l21_tau_min * t;  % 线性退火
        if tau_now > 0
            row_norm = sqrt(sum(P.^2, 2)) + 1e-12;       % d x 1
            shrink   = max(0, 1 - tau_now ./ row_norm);  % d x 1
            P = bsxfun(@times, P, shrink);
        end
    end

    % 可选：列稀疏（top_s）保持兼容
    top_s = gparam(HyperPara,'top_s',0);
    if top_s > 0
        for c = 1:size(P,2)
            v = P(:,c);
            [~,ix] = maxk(abs(v), min(top_s,numel(v)));
            vv = zeros(size(v)); vv(ix) = v(ix);
            nv = norm(vv);
            if nv > 0, P(:,c) = vv / nv; else, P(:,c) = vv; end
        end
    end

    % (I) 目标值
    numJ = trace(P' * B * P);
    denJ = trace(P' * C * P) + 1e-12;
    J = numJ / denJ;

    % (J) 更新 W（显式特征置信度）
    W_new = compute_feature_confidence(X, F, k, w_pow_mad, w_pow_nn, w_pow_rel);
    t   = it / maxIter;
    etaW = etaW0 * (1 - t) + etaW_min * t; etaW = max(0,min(1,etaW));
    W = (1 - etaW) * W + etaW * W_new;
    W = W / max(max(W),1e-12);
    W = max(w_min, min(1.0, W));

    % (K) 每 Kref 次迭代更新 F（W 引导的相似度传播）
    if Kref <= 1 || mod(it, Kref) == 0
        Xz = X * P;
        [knn_idx,~] = knnsearch(Xz, Xz, 'K', min(k+1,n)); knn_idx = knn_idx(:,2:end);

        tauZ = robust_scale(Xz);
        Xw   = X .* (sqrt(W(:)'+1e-12));
        tauW = robust_scale(Xw);

        F_new = zeros(n,q);
        for i = 1:n
            nei = knn_idx(i,:);
            dz2 = sum((Xz(i,:) - Xz(nei,:)).^2, 2);
            wz  = exp(-dz2 / (2*(tauZ^2 + 1e-12)));
            diffw = Xw(nei,:) - Xw(i,:); dw2 = sum(diffw.^2, 2);
            ww  = exp(-dw2 / (2*(tauW^2 + 1e-12)));
            wnei = wz .* ww;    % 融合两个空间
            wnei = wnei / (sum(wnei) + 1e-12);

            agg = (wnei' * F(nei,:));  % 1 x q
            cand = find(Y_partial(i,:)==1);
            if isempty(cand)
                [~,mx] = max(agg); F_new(i,mx)=1; continue;
            end
            svec = zeros(1,q);
            svec(cand) = softmax_masked(agg(cand), T);
            if max(svec) > rho_fix
                [~,mx] = max(svec); F_new(i,:)=0; F_new(i,mx)=1;
            else
                F_new(i,:) = svec;
            end
        end
        etaF_now = etaF0 * (1 - t) + etaF_min * t; etaF_now = max(0,min(1,etaF_now));
        F = (1 - etaF_now) * F + etaF_now * F_new;
        F = F ./ (sum(F,2) + 1e-12);
    end

    % (L) alpha 衰减
    alpha = alpha * decay_s + (1-decay_s) * alpha_floor;
    alpha = max(alpha_min, min(alpha_max, alpha));

    [PP,~]  = qr(P,0); [PP0,~] = qr(P_old,0);
    sub_diff = norm(PP*PP' - PP0*PP0','fro') / (norm(PP0*PP0','fro') + 1e-12);
   

    J_hist(end+1) = J;
    if numel(J_hist) > patience
        J_old = J_hist(end - patience);
        J_new = J_hist(end);
        if (J_new - J_old) <= eps_rel * max(1, abs(J_old))
                        break;
        end
    end
    if it>1 && sub_diff < minMargin
                break;
    end
    P_old = P;
end

P_final = P;
fprintf('  [NF-PMLD(+rank,+l21)] done.\n');
end

% ---------------- helpers ----------------
function v = gparam(S,name,defaultValue)
    if exist('S','var') && isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        v = S.(name);
    else
        v = defaultValue;
    end
end

function L = build_pl_graph_anchor_local(X, Yp, m, kz, perp, k_graph)
    n = size(X,1);
    try
        [~,C] = kmeans(X,m,'MaxIter',100,'Replicates',3,'OnlinePhase','on');
    catch
        rng(1); C = X(randsample(n,m),:);
    end
    [idx,dist] = knnsearch(C, X, 'K', min(kz,m));
    Z = sparse(n,m);
    for i=1:n
        di = dist(i,:) - min(dist(i,:));
        sigma = perplexity_to_sigma(di, perp);
        wi = exp(-di./(sigma+1e-12)); wi = wi./(sum(wi)+1e-12);
        Z(i, idx(i,:)) = wi;
    end
    mEff = size(Z,2);
    Wc = zeros(mEff,mEff);
    [ia,~] = knnsearch(C, C, 'K', min(6,mEff)); ia = ia(:,2:end);
    ymask = (sum(Yp,2)>0);
    for a=1:mEff
        na = ia(a,:);
        Sa = (Z(:,a)>0);
        for t=1:numel(na)
            b = na(t);
            Sb = (Z(:,b)>0);
            inter = sum((Sa & Sb) & ymask);
            uni   = sum((Sa | Sb) & ymask);
            if uni>0
                w = inter/uni; Wc(a,b)=w; Wc(b,a)=w;
            end
        end
    end
    Dz = diag(sum(Wc,2)); Lz = Dz - Wc;
    L = Z * Lz * Z'; L = 0.5*(L+L');
    trLX = trace((X')*L*X);
    if trLX > 0, L = L * (n / trLX); end
    [nn_idx,~] = knnsearch(X, X, 'K', min(k_graph+1, n)); nn_idx = nn_idx(:,2:end);
    A = sparse(n,n);
    for i=1:n, A(i, nn_idx(i,:)) = 1; end
    A = A | A';
    L = L .* A;
    dvec = sum(L,2);
    L = spdiags(dvec,0,n,n) - L;
end

function Lg = build_global_graph_cosine(X, k)
    n = size(X,1);
    Xn = X ./ (sqrt(sum(X.^2,2))+1e-12);
    S  = zeros(n,k);
    IDX = zeros(n,k);
    [idx0,~] = knnsearch(Xn, Xn, 'K', min(k+1,n)); idx0 = idx0(:,2:end);
    for i=1:n
        cand = idx0(i,:);
        cs   = Xn(i,:)*Xn(cand,:)';
        [cs_sorted, ord] = sort(cs, 'descend');
        ki = min(k, numel(cand));
        IDX(i,1:ki) = cand(ord(1:ki));
        S(i,1:ki)   = max(0, cs_sorted(1:ki));
    end
    W = sparse(repmat((1:n)',1,k), IDX, S, n, n);
    W = max(W, W');
    D = spdiags(sum(W,2),0,n,n);
    Lg = D - W;
end

function sigma = perplexity_to_sigma(d, perp)
    if all(d==0), sigma=1e-3; return; end
    lo=1e-4; hi=1e2; target=log(perp+1e-12);
    for it=1:25
        mid=(lo+hi)/2;
        p=exp(-d./mid); p=p/(sum(p)+1e-12);
        H=-sum(p.*log(p+1e-12));
        if H>target, hi=mid; else, lo=mid; end
    end
    sigma=(lo+hi)/2;
end

function tau = robust_scale(X)
    n = size(X,1);
    m = min(2000, n*(n-1)/2);
    if m <= 0, tau = 1.0; return; end
    ii = randi(n, m, 1); jj = randi(n, m, 1);
    keep = ii~=jj; ii = ii(keep); jj = jj(keep);
    if isempty(ii), tau = 1.0; return; end
    diff = X(ii,:) - X(jj,:);
    dist = sqrt(sum(diff.^2,2));
    med  = median(dist);
    mad  = median(abs(dist - med)) * 1.4826;
    tau = max(1e-3, med + mad);
end

function W = compute_feature_confidence(X, F, k, pow_mad, pow_nn, pow_rel)
    [n,d] = size(X);
    med  = median(X,1);
    madv = median(abs(X - med),1) + 1e-12;
    w_mad = 1 ./ madv; w_mad = w_mad / (max(w_mad)+1e-12);

    [idx,~] = knnsearch(X, X, 'K', min(k+1,n)); idx = idx(:,2:end);
    res = zeros(1,d);
    for i=1:n
        nei = idx(i,:); mu = mean(X(nei,:),1);
        res = res + abs(X(i,:) - mu);
    end
    res = res/n + 1e-12;
    w_nn = 1 ./ res; w_nn = w_nn / (max(w_nn)+1e-12);

    sumF = sum(F,1);
    mu   = mean(X,1);
    mu_j = (F' * X) ./ (sumF'+1e-12);
    sb_per_dim = sum( bsxfun(@times, (mu_j - mu).^2, sumF'), 1 );
    sw_per_dim = zeros(1,d);
    for j = 1:size(F,2)
        if sumF(j) < 1e-12, continue; end
        xj = X; wj = F(:,j);
        mj = mu_j(j,:);
        sw_per_dim = sw_per_dim + (wj' * ((xj - mj).^2)) ;
    end
    rel = sb_per_dim ./ (sw_per_dim + 1e-12);
    rel = rel / (max(rel)+1e-12);
    w_rel = rel;

    W = (w_mad.^pow_mad) .* (w_nn.^pow_nn) .* (w_rel.^pow_rel);
    W = W(:);
    if max(W)<=0, W = ones(d,1); end
    W = W / (max(W)+1e-12);
end

function s = softmax_masked(z, T)
    z = z - max(z);
    e = exp(z / max(T,1e-3));
    s = e / (sum(e) + 1e-12);
end

function p = percentile(v, pct)
    if isempty(v), p = 0; return; end
    v = sort(v(:));
    pct = max(0, min(100, pct));
    if numel(v) == 1, p = v; return; end
    idx = 1 + (pct/100) * (numel(v) - 1);
    lo = floor(idx); hi = ceil(idx);
    if lo == hi
        p = v(lo);
    else
        w = idx - lo; p = (1 - w) * v(lo) + w * v(hi);
    end
end

function B_rank = build_B_rank(mu_j, F, rank_topk)
    [d,q] = size(mu_j);
    Wpair = zeros(q,q);  % 有向计数

    n = size(F,1);
    if rank_topk > 0
        K = min(rank_topk, q);
        for i = 1:n
            [~,ord] = sort(F(i,:), 'descend');
            idx = ord(1:K);
            fi  = F(i,idx);
            D   = fi(:) - fi(:)';          % K x K
            Dp  = max(D, 0);               % 只统计正向 (a>b)
            Wpair(idx,idx) = Wpair(idx,idx) + Dp;
        end
    else
        % 全量
        for i = 1:n
            fi = F(i,:);
            D  = fi(:) - fi(:)';           % q x q
            Dp = max(D, 0);
            Wpair = Wpair + Dp;
        end
    end

    S = abs(Wpair - Wpair');   % 无向 pair 权重
    % 累积到 B_rank
    B_rank = zeros(d,d);
    for a = 1:q-1
        va = mu_j(:,a);
        for b = a+1:q
            vb = mu_j(:,b);
            w_ab = S(a,b);
            if w_ab > 0
                diff = (va - vb);
                B_rank = B_rank + w_ab * (diff * diff');
            end
        end
    end
    % 数值健壮性
    B_rank = (B_rank + B_rank')/2 + 1e-12*eye(d);
end
