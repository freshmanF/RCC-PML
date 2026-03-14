
clc; clear; close all;

xlsx_out = './ALL_RESULTS_testa.xlsx';
if exist(xlsx_out,'file'), delete(xlsx_out); end
header = {'Dataset','avgCLs',...
          'HammingLoss','MicroF1','AP','OneError','RankingLoss','Coverage',...
          'BestDim','BestAP'};
writecell(header, xlsx_out, 'Sheet','Results', 'Range','A1');
row_ptr = 2;


cfg = struct(   'exampledata',        [4 5 6]);

datasets = fieldnames(cfg);


feaNoiseRate = 0.3;         
rng(0);               

for di = 1:numel(datasets)
    ds = datasets{di};
    ks = cfg.(ds);

    for ki = 1:numel(ks)
        data_name = ['./data/' ds '.mat']; 
        load(data_name);
        rng(0); 
        avgCLs = ks(ki);                  

        fprintf('数据集: %s, avgCLs率: %.1f, 特征噪声率: %.1f\n', data_name, avgCLs, feaNoiseRate);
        [train_data, settings] = mapminmax(train_data');
        test_data = mapminmax('apply', test_data', settings);
        train_data = train_data';
        test_data = test_data';
        train_data(isnan(train_data)) = 0;
        test_data(isnan(test_data)) = 0;

        train_target(train_target==-1) = 0;
        test_target(test_target==-1) = 0;

        fprintf('正在生成PML标签...\n');
        PL_transposed = getPartialLabel_fixed_k(train_target, avgCLs); 
        PL = PL_transposed'; % n x q
        [num_instance, num_feature] = size(train_data);

        fprintf('正在添加特征噪声...\n');
        clear_data = [train_data; test_data];
        noisy_data = FeatureNoise(clear_data, feaNoiseRate);
        train_data = noisy_data(1:num_instance, :);
        test_data = noisy_data(num_instance+1:end, :);
        fprintf('数据准备完成。\n');


        model_name = 'RCC-PML';

        HyperPara.decay_s = 0.95;
        HyperPara.k = 10;
        HyperPara.maxIter = 50;
        HyperPara.k_graph=10;
        HyperPara.eta_local=4e-4;
        HyperPara.eta_global=2e-4;
        HyperPara.lambda_w=2e-3;
        HyperPara.d_prime = ceil(num_feature*0.5);

        HyperPara.beta_rank = 1e-3;
        HyperPara.pair_sample = 20000;
        HyperPara.F_update_period = 2;
        HyperPara.F_momentum = 0.3;
        HyperPara.F_momentum_min = 0.12;
        HyperPara.F_entropy_low = 0.40;
        HyperPara.F_delta_low   = 4e-3;
        HyperPara.alpha_fast    = 0.10;
        HyperPara.alpha_slow    = 0.03;
        HyperPara.alpha_min = 0.3; HyperPara.alpha_max = 0.8; HyperPara.alpha_floor = 0.3;
        HyperPara.eta = 4e-4;
        HyperPara.rank_jaccard_thresh = 0.15;





        fprintf('--- 运行 %s 算法学习投影矩阵 ---\n', model_name);
        tic;
        P = learn_projection_RCC_PML(train_data, PL, HyperPara);
        time_learn_p = toc;
        fprintf('学习投影矩阵 P 完成，耗时: %.2f 秒。\n', time_learn_p);

        Num = 10; Smooth = 1; 
        PL_eval = PL;
        PL_eval(PL_eval==0) = -1;

        if size(test_target, 2) < size(test_target, 1)
            test_target = test_target';
        end

       
            NumOfInterval = 25;
            Dt = HyperPara.d_prime;
            if Dt < NumOfInterval 
                step = 1;
                NumOfInterval = Dt;
            else
                step = floor(Dt / NumOfInterval);
            end

            iterResult = zeros(15, NumOfInterval);
            dims_to_test = 1:step:Dt; 

            for i = 1:length(dims_to_test)
                d_prime = dims_to_test(i); 

                P_current = P(:, 1:d_prime);
                train_data_transformed = train_data * P_current;
                test_data_transformed  = test_data  * P_current;

                [Prior,PriorN,Cond,CondN] = MLKNN_train(train_data_transformed, PL_eval', Num, Smooth);
                [~,~,~,~,~,~,Outputs,Pre_Labels] = MLKNN_test(train_data_transformed, PL_eval', test_data_transformed, test_target, Num, Prior, PriorN, Cond, CondN);

                pred_pos_per_inst = mean(sum(Pre_Labels==1, 1));
                true_pos_per_inst = mean(sum(test_target==1, 1));
                fprintf('[card-calib] before: avg(pred #pos)=%.2f | avg(true #pos)=%.2f | q=%d\n', ...
                    pred_pos_per_inst, true_pos_per_inst, size(test_target,1));

                Kbar = max(1, round(mean(sum(train_target==1, 1))));
                Scores = Outputs;

                [q_lab, n_te] = size(Scores);
                [S_sorted, ord] = sort(Scores, 1, 'descend');
                Pre_cal = -ones(q_lab, n_te);

                gamma_rel = 0.15;
                peak_gain = 1.5;
                for i_col = 1:n_te
                    scol = S_sorted(:, i_col);
                    k = Kbar;

                    if q_lab > (Kbar + 1)
                        sstd = std(scol);
                        sstd = max(sstd, 1e-8);
                        gap = scol(Kbar) - scol(Kbar+1);
                        rel = gap / sstd;

                        if rel < gamma_rel
                            k = min(Kbar + 1, q_lab);
                        elseif rel > peak_gain * gamma_rel && Kbar > 1
                            k = Kbar - 1;
                        else
                            k = Kbar;
                        end
                    end

                    keep = ord(1:k, i_col);
                    Pre_cal(keep, i_col) = 1;
                end

                Pre_Labels = Pre_cal;

                pred_pos_per_inst2 = mean(sum(Pre_Labels==1, 1));
                fprintf('[card-calib]  after: avg(pred #pos)=%.2f (target≈%.2f)\n', ...
                    pred_pos_per_inst2, Kbar);

                test_target_eval = test_target;
                test_target_eval(test_target_eval==0) = -1;
                pred_pos_per_inst = mean(sum(Pre_Labels == 1, 1));
                true_pos_per_inst = mean(sum(test_target == 1, 1));
                fprintf('[sanity] avg(pred #pos)=%.2f | avg(true #pos)=%.2f | q=%d\n', ...
                    pred_pos_per_inst, true_pos_per_inst, size(test_target,1));

                tmpResult = EvaluationAll(Pre_Labels, Outputs, test_target_eval);
                iterResult(:, i) = tmpResult;
                fprintf('评估完成: %d / %d (新维度: %d)\n', i, length(dims_to_test), d_prime);
            end
       

        Avg_Result = [mean(iterResult,2), std(iterResult,1,2)];
        x_axis = dims_to_test;
      
        
            [bestAP, ap_idx] = max(iterResult(12,:));
            best_dim = x_axis(ap_idx);

            P_best   = P(:, 1:best_dim);
            Ztr_best = train_data * P_best;
            Zte_best = test_data  * P_best;

            [Prior,PriorN,Cond,CondN] = MLKNN_train(Ztr_best, PL_eval', Num, Smooth);
            [~,~,~,~,~,~,Outputs_best,Pre_Labels_best] = MLKNN_test(Ztr_best, PL_eval', Zte_best, test_target, Num, Prior, PriorN, Cond, CondN);

            Kbar = max(1, round(mean(sum(train_target==1, 1))));
            Scores = Outputs_best;
            [q_lab, n_te] = size(Scores);
            [S_sorted, ord] = sort(Scores, 1, 'descend');
            Pre_cal = -ones(q_lab, n_te);
            gamma_rel = 0.15; peak_gain = 1.5;
            for i_col = 1:n_te
                scol = S_sorted(:, i_col);
                k = Kbar;
                if q_lab > (Kbar + 1)
                    sstd = std(scol); sstd = max(sstd, 1e-8);
                    gap = scol(Kbar) - scol(Kbar+1);
                    rel = gap / sstd;
                    if rel < gamma_rel
                        k = min(Kbar + 1, q_lab);
                    elseif rel > peak_gain * gamma_rel && Kbar > 1
                        k = Kbar - 1;
                    end
                end
                keep = ord(1:k, i_col);
                Pre_cal(keep, i_col) = 1;
            end
            Pre_Labels_best = Pre_cal;

            test_target_eval = test_target;
            test_target_eval(test_target_eval==0) = -1;
            finalResult = EvaluationAll(Pre_Labels_best, Outputs_best, test_target_eval);  % 15x1

            ix_Hamming  = 1;
            ix_MicroF1  = 11;
            ix_AP       = 12;
            ix_OneError = 13;
            ix_RankLoss = 14;
            ix_Coverage = 15;

            fprintf('\n=== Final 6 metrics  ===\n');
            fprintf('Hamming Loss (lower better)         : %.4f\n', finalResult(ix_Hamming));
            fprintf('Micro-F1 Measure (higher better)    : %.4f\n', finalResult(ix_MicroF1));
            fprintf('Average Precision (higher better)   : %.4f\n', finalResult(ix_AP));
            fprintf('One Error (lower better)            : %.4f\n', finalResult(ix_OneError));
            fprintf('Ranking Loss (lower better)         : %.4f\n', finalResult(ix_RankLoss));
            fprintf('Coverage (lower better)             : %.4f\n', finalResult(ix_Coverage));
        end

        row = {ds, avgCLs, ...
               finalResult(ix_Hamming), finalResult(ix_MicroF1), finalResult(ix_AP), ...
               finalResult(ix_OneError), finalResult(ix_RankLoss), finalResult(ix_Coverage), ...
               best_dim, bestAP};
        writecell(row, xlsx_out, 'Sheet','Results', 'Range', sprintf('A%d', row_ptr));
        row_ptr = row_ptr + 1;

    end


fprintf('\nAll done. Results saved to: %s\n', xlsx_out);