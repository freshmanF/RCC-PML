function HammingLoss = Hamming_loss(Pre_Labels, test_target)
% Computing the Hamming loss (robust to {-1,1} or {0,1} encodings)
% Pre_Labels : L x N predicted labels (each entry in {-1,1} or {0,1})
% test_target: L x N ground-truth labels (each entry in {-1,1} or {0,1})

    % --- normalize both to 0/1 --- 
    % If input already 0/1, this keeps it; if -1/1, maps -1->0, +1->1
    P = Pre_Labels;
    T = test_target;

    % handle NaN gracefully
    P(isnan(P)) = 0; 
    T(isnan(T)) = 0;

    % detect encoding by value range quickly
    % if any value < 0, assume {-1,1} and map via >0
    if any(P(:) < 0), P = (P > 0); end
    if any(T(:) < 0), T = (T > 0); end

    % ensure logical/uint8 to speed xor
    P = P ~= 0;
    T = T ~= 0;

    % Hamming loss = average per (label,instance) XOR mismatch
    miss_pairs = nnz(xor(P, T));
    [num_class, num_instance] = size(P);
    HammingLoss = miss_pairs / (num_class * num_instance);
end
