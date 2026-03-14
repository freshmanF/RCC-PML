function [Y_partial] = getPartialLabel_fixed_k(Y_gt, p)
% =========================================================================
%                  为PML生成噪声标签 (固定数量法)
% =========================================================================
% 功能:
% 遵循学术界标准做法，为每个样本添加固定数量的噪声标签，使其
% 候选标签总数达到一个预设的目标值 p。
%
% 输入:
%   Y_gt: q x n 的原始真实标签矩阵 (q: 标签数, n: 样本数)
%   p:    目标候选标签数量 (一个整数, 例如 3, 5, 7)
%
% 输出:
%   Y_partial: q x n 的部分多标签矩阵，其中每列的非零元素数量约为 p
% =========================================================================

fprintf('--- 正在使用“固定数量法”生成PML标签, 目标候选标签数 p = %d ---\n', p);

% --- 维度检查与准备 ---
[q, n] = size(Y_gt); % 获取标签数和样本数

% 为了方便按样本（行）处理，先将标签矩阵转置为 n x q
Y_gt_proc = Y_gt'; 
% 初始化部分多标签矩阵，初始状态等于真实标签矩阵
Y_partial_proc = Y_gt_proc;

num_labels_added_total = 0; % 用于统计总共添加了多少噪声标签

% --- 遍历每一个样本进行处理 ---
for i = 1:n
    instance_labels = Y_gt_proc(i, :); % 获取当前样本的标签行向量
    
    % 找到真实标签(1)和非相关标签(0)的索引
    true_label_indices = find(instance_labels == 1);
    false_label_indices = find(instance_labels == 0);
    
    % 计算当前样本已有的真实标签数量
    num_gt = length(true_label_indices);
    
    if num_gt >= p
        % 如果真实标签数量已经达到或超过目标值p，则不添加任何噪声标签
        % 该样本的候选标签集就是其真实标签集
        continue;
    else
        % 需要添加的噪声标签数量
        num_to_add = p - num_gt;
        
        % 检查可供选择的噪声标签是否足够
        num_available_false = length(false_label_indices);
        if num_to_add > num_available_false
            % 如果不够，则将所有非相关标签都变成噪声标签
            num_to_add = num_available_false;
        end
        
        if num_to_add > 0
            % 随机打乱非相关标签的索引
            shuffled_false_indices = false_label_indices(randperm(num_available_false));
            
            % 选取需要添加的噪声标签的索引
            indices_to_flip = shuffled_false_indices(1:num_to_add);
            
            % 将这些位置的标签从0翻转为1
            Y_partial_proc(i, indices_to_flip) = 1;
            
            num_labels_added_total = num_labels_added_total + num_to_add;
        end
    end
end

% --- 将结果转置回原始的 q x n 格式 ---
Y_partial = Y_partial_proc';

% --- 输出统计信息 ---
avg_gls = sum(Y_gt(:)) / n;
avg_cls = sum(Y_partial(:)) / n;
fprintf('生成完成。\n');
fprintf('原始平均真实标签数 (avg.#GLs): %.2f\n', avg_gls);
fprintf('生成后平均候选标签数 (avg.#CLs): %.2f\n', avg_cls);
fprintf('总共添加了 %d 个噪声标签。\n', num_labels_added_total);

end