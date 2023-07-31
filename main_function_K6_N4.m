function [achieved, prob_rate, prob_delay, average_delay, time, total_achieved, total_delay, nb_JSub6, nb_JmmW, optimal_val_sub6, optimal_val_mmW] = main_function(partition, filename, iter_epoch, iter_learning)

    tic;

    global l beta J_max schedule_frameTime max_epoch N_sub6 N_mmW bandwidth_s6 bandwidth_mmW band_per_RB


    l = 160; 
    beta = 3200;
    J_max = 10;
    schedule_frameTime = 5;
    max_epoch = 10;
    N_sub6 = 2;
    N_mmW = 2;

    bandwidth_s6 = 1e+8; % Bandwidth sub6 =  1e8 Hz = 100 MHz
    bandwidth_mmW = 1e+9; %Bandwidth mmWave = 1e9 Hz = 1 GHz
    band_per_RB = bandwidth_s6 / N_sub6; 

    
    %%%Non LoS users already computed into SINR_mmW_central_users file
    % Files with the channels values, the LoS proba are already computed
    % into these files
    load('MATLAB/ressources/topologies/topo_3x3/6users/Same_Channels_K6_nLoS_2.mat', 'SINR_mmW_value');
    load('MATLAB/ressources/topologies/topo_3x3/6users/Same_Channels_K6_nLoS_2.mat', 'SINR_sub6_value');

    SINR_sub6_allframes = SINR_sub6_value;
    SINR_mmW_allframes = SINR_mmW_value;

    loader = load(filename);
    los = loader.('los_central_users');
    users_infos(1,:) = los;    
    
    load('MATLAB/ressources/users_infos_K6_2.mat', 'users_infos');

    
    [achieved, prob_rate, prob_delay, average_delay, time, total_achieved, total_delay, nb_JSub6, nb_JmmW, optimal_val_sub6, optimal_val_mmW]  = embeded_scheduler(partition, iter_epoch, iter_learning);

    function [achieved, prob_rates, prob_delay, average_delay, time, total_achieved, total_delay, nb_JSub6, nb_JmmW, optimal_val_sub6, optimal_val_mmW] = embeded_scheduler(partition, iter_epoch, iter_learning)

        nb_JSub6 = zeros(1, max_epoch);
        nb_JmmW = zeros(1, max_epoch);
        optimal_val_mmW = zeros(1, max_epoch);
        optimal_val_sub6 = zeros(1, max_epoch);
        
        %Create subgroups
        sub6_users = partition == 1;
        mmW_users = partition == 2;
        unpartitioned_users = partition == 0;
        
        K_total = length(partition);
        K_sub6 = sum(sub6_users);
        K_mmW = sum(mmW_users);
        
        %manipulate loaded SINR matrix to only keep for Sub6 users and
        %sample MaxEpoch frames
        SINR_sub6 = SINR_sub6_allframes(:, sub6_users, iter_learning, iter_epoch);
        
        %manipulate loaded SINR matrix to only keep for mmW users and
        %sample MaxEpoch frames
        SINR_mmW = SINR_mmW_allframes(:,mmW_users, iter_learning, iter_epoch);
        
        %Create all the informations about the subgroup necessary for the
        %scheduler
        users_infos_sub6 = users_infos([1 2 6 4], sub6_users); %los, rate_req, error_req, delay_req
        Q_inv_sub6 = sqrt(2)*erfinv(1 - 2*users_infos_sub6(3,:))';
        
        users_infos_mmW = users_infos([1 2 6 4], mmW_users);
        Q_inv_mmW = sqrt(2)*erfinv(1 - 2*users_infos_mmW(3,:))';
        users_infos_temp = users_infos_mmW;
        
        
        %initialize the mean achieved rate and the probability arrays for
        %the Reward
        achieved_means = zeros(1, K_total);
        successfulRate_epoch = zeros(1, K_total);
        failDelay_epoch = zeros(1, K_total);
        average_delay = zeros(1, K_total);
        delay_means = zeros(1, K_total);
        
        total_achieved = zeros(max_epoch, K_total);
        total_delay = zeros(max_epoch, K_total);
        
        for i = 1:max_epoch
            
            previousFval_sub6 = 0;
            previousFval_mmW = 0;
            
            SINR_sub6_epoch = SINR_sub6(i,:); %Toute la ligne i
            SINR_mmW_epoch = SINR_mmW(i,:);
                        
            %Initial point for the convex optimize
            %schedule_init_sub6= ones(N_sub6, schedule_frameTime, K_sub6);
            %schedule_init_mmW = ones(N_mmW, schedule_frameTime, K_mmW);
            schedule_init_sub6 = init_schedule2(N_sub6, K_sub6, schedule_frameTime);
            schedule_init_mmW =  init_schedule2(N_mmW, K_mmW, schedule_frameTime);
            
            sched_0_sub6 = schedule_init_sub6;
            sched_0_mmW = schedule_init_mmW;
            
            %fmincon parameters creation: function to opti, lower & upper bounds, non
            %linear constraints
            options = optimoptions(@fmincon,'Algorithm','sqp', 'Display', 'off');
            %options2 = optimoptions(@fmincon,'Algorithm','sqp', 'Display', 'iter-detailed');
            
            lb_sub6 = zeros(N_sub6, schedule_frameTime, K_sub6);
            ub_sub6 = ones(N_sub6, schedule_frameTime, K_sub6);
            lb_mmW = zeros(N_mmW, schedule_frameTime, K_mmW);
            ub_mmW = ones(N_mmW, schedule_frameTime, K_mmW);
            
            fun_sub6 = @(sched) U_bar(sched, schedule_init_sub6, SINR_sub6_epoch, Q_inv_sub6, l, beta, band_per_RB, schedule_frameTime, N_sub6, K_sub6);
            cons_sub6 = @(sched) allcons(sched, schedule_init_sub6, SINR_sub6_epoch, Q_inv_sub6, l, users_infos_sub6, K_sub6, N_sub6, band_per_RB, schedule_frameTime);
            
            fun_mmW = @(sched) U_bar(sched, schedule_init_mmW, SINR_mmW_epoch, Q_inv_mmW, l, beta, bandwidth_mmW, schedule_frameTime, N_mmW, K_mmW);
            cons_mmW = @(sched) allcons(sched, schedule_init_mmW, SINR_mmW_epoch, Q_inv_mmW, l, users_infos_mmW, K_mmW, N_mmW, bandwidth_mmW, schedule_frameTime);
            

            for j = 1:J_max
                if K_sub6 ~= 0
                    %[x_sub6, fval_sub6, exitflag_sub6, output_sub6] = fmincon(fun_sub6, sched_0_sub6, [], [], [], [], lb_sub6, ub_sub6, cons_sub6, options);
                    [x_sub6, fval_sub6, exitflag_sub6, output_sub6] = fmincon(fun_sub6, schedule_init_sub6, [], [], [], [], lb_sub6, ub_sub6, cons_sub6, options);
                    
                    schedule_init_sub6 = x_sub6;
                    
                    %%Stop if local optimum reached
                    if abs(previousFval_sub6 - fval_sub6) <= 0.01
                        nb_JSub6(i) = j;
                        K_sub6 = 0;
                    end
                    previousFval_sub6 = fval_sub6;
                    
                    if j ==10
                        nb_JSub6(i) = j;
                    end
                                        
                    fun_sub6 = @(sched) U_bar(sched, schedule_init_sub6, SINR_sub6_epoch, Q_inv_sub6, l, beta, band_per_RB, schedule_frameTime, N_sub6, K_sub6);
                    cons_sub6 = @(sched) allcons(sched, schedule_init_sub6, SINR_sub6_epoch, Q_inv_sub6, l, users_infos_sub6, K_sub6, N_sub6, band_per_RB, schedule_frameTime);
                end
                
                if K_mmW ~= 0
                    %[x_mmW,fval_mmW,exitflag_mmW,output_mmW] = fmincon(fun_mmW, sched_0_mmW, [], [], [], [], lb_mmW, ub_mmW, cons_mmW, options);
                    [x_mmW,fval_mmW,exitflag_mmW,output_mmW] = fmincon(fun_mmW, schedule_init_mmW, [], [], [], [], lb_mmW, ub_mmW, cons_mmW, options);

                    schedule_init_mmW = x_mmW;
                    
                    %%Stop if local optimum reached
                    if abs(previousFval_mmW - fval_mmW) <= 0.01
                        nb_JmmW(i) = j;
                        K_mmW = 0;
                    end
                    previousFval_mmW = fval_mmW;
                    
                    if j == 10
                        nb_JmmW(i) = j;
                    end
                                        
                    fun_mmW = @(sched) U_bar(sched, schedule_init_mmW, SINR_mmW_epoch, Q_inv_mmW, l, beta, bandwidth_mmW, schedule_frameTime, N_mmW, K_mmW);
                    cons_mmW = @(sched) allcons(sched, schedule_init_mmW, SINR_mmW_epoch, Q_inv_mmW, l, users_infos_mmW, K_mmW, N_mmW, bandwidth_mmW, schedule_frameTime);
                end
            end
            
            
            current_achieved = zeros(1, K_total);
            current_delay = zeros(1, K_total);
            
            K_sub6 = sum(sub6_users);
            K_mmW = sum(mmW_users);
            
            if K_sub6 ~= 0
                
                %Retrieve the solution schedule for this epoch
                x_sub6(x_sub6 < 0.5) = 0;
                x_sub6(x_sub6 > 0.5) = 1;
                
                optimal_schedule_sub6 = x_sub6;
                
                userperblock_sub6 = (sum(optimal_schedule_sub6, 3) - 1) <= 0;
                maxblock_sub6 = (sumcol(optimal_schedule_sub6) - N_sub6) <= 0;
                
                %                 if exitflag_sub6 > 0 % Optim OK
                %                     if any(~userperblock_sub6, 'all') == true
                %                         cptx = cptx + 1;
                %                     end
                %                     %compute the achieved rate for this epoch's solution
                %                     cap_sub6 = F_k(optimal_schedule_sub6, SINR_sub6_epoch, band_per_RB, schedule_frameTime, N_sub6, K_sub6);
                %                     disp_sub6 = V_k(optimal_schedule_sub6, SINR_sub6_epoch, Q_inv_sub6, l, band_per_RB, schedule_frameTime, N_sub6, K_sub6);
                %
                %                     current_achieved(sub6_users) = cap_sub6 - disp_sub6;
                %
                %                     %Compute the achieved delay for this epoch's solution
                %                     current_delay(sub6_users) = calcul_delay(optimal_schedule_sub6, K_sub6, schedule_frameTime);
                %
                
                if userperblock_sub6 & maxblock_sub6
                    
                    optimal_val_sub6(i) = fval_sub6;
                    %compute the achieved rate for this epoch's solution
                    cap_sub6 = F_k(optimal_schedule_sub6, SINR_sub6_epoch, band_per_RB, schedule_frameTime, N_sub6, K_sub6);
                    disp_sub6 = V_k(optimal_schedule_sub6, SINR_sub6_epoch, Q_inv_sub6, l, band_per_RB, schedule_frameTime, N_sub6, K_sub6);
                    
                    current_achieved(sub6_users) = cap_sub6 - disp_sub6;
                    
                    %Compute the achieved delay for this epoch's solution
                    current_delay(sub6_users) = calcul_delay(optimal_schedule_sub6, K_sub6, schedule_frameTime);
                else
                    current_achieved(sub6_users) = 0;
                    current_delay(sub6_users) = schedule_frameTime + 1;
                end
            end
            
            if K_mmW ~= 0
                
                
                %Retrieve the solution schedule for this epoch
                x_mmW(x_mmW < 0.5) = 0;
                x_mmW(x_mmW > 0.5) = 1;
                
                optimal_schedule_mmW = x_mmW;
                
                userperblock_mmW = (sum(optimal_schedule_mmW, 3) - 1) <= 0;
                maxblock_mmW = (sumcol(optimal_schedule_mmW) - N_mmW) <= 0;
                
                %                 if exitflag_mmW > 0 % Optim OK
                %                     if any(~userperblock_mmW, 'all') == true
                %                         cpty = cpty + 1;
                %                     end
                %                     %compute the achieved rate for this epoch's solution
                %                     cap_mmW = F_k(optimal_schedule_mmW, SINR_mmW_epoch, bandwidth_mmW, schedule_frameTime, N_mmW, K_mmW);
                %                     disp_mmW = V_k(optimal_schedule_mmW, SINR_mmW_epoch, Q_inv_mmW, l, bandwidth_mmW, schedule_frameTime, N_mmW, K_mmW);
                %
                %                     current_achieved(mmW_users) = cap_mmW - disp_mmW;
                %
                %                     %Compute the achieved delay for this epoch's solution
                %                     current_delay(mmW_users) = calcul_delay(optimal_schedule_mmW, K_mmW, schedule_frameTime);
                
                if userperblock_mmW & maxblock_mmW %Optim not ok because Rate or Delay not OK for at least one user
                    
                    optimal_val_mmW(i) = fval_mmW;
                    %compute the achieved rate for this epoch's solution
                    cap_mmW = F_k(optimal_schedule_mmW, SINR_mmW_epoch, bandwidth_mmW, schedule_frameTime, N_mmW, K_mmW);
                    disp_mmW = V_k(optimal_schedule_mmW, SINR_mmW_epoch, Q_inv_mmW, l, bandwidth_mmW, schedule_frameTime, N_mmW, K_mmW);
                    
                    current_achieved(mmW_users) = cap_mmW - disp_mmW;
                    
                    %Compute the achieved delay for this epoch's solution
                    current_delay(mmW_users) = calcul_delay(optimal_schedule_mmW, K_mmW, schedule_frameTime);
                else
                    current_achieved(mmW_users) = 0;
                    current_delay(mmW_users) = schedule_frameTime + 1;
                end
            end
            
            %Update Delay means
            
            %%If the delay equals 0, the user is not assigned to any sub
            %%freq. or he has no rb allocated. => I pput a big delay for
            %%these users
            current_delay(current_delay == 0) = schedule_frameTime + 1;
            
            delay_means = delay_means + (current_delay - delay_means)./i;
            
            %update the mean achieved rate
            achieved_means = achieved_means + (current_achieved - achieved_means)./i;
            successfulRate_epoch = successfulRate_epoch + (achieved_means < users_infos(2,:));
            failDelay_epoch = failDelay_epoch + (current_delay > users_infos(4,:));
            
            total_achieved(i,:) = current_achieved;
            total_delay(i,:) = current_delay;
            
            %update the user requirements knowing their mean achieved rate so
            %far
            users_infos_sub6(2,:) = (i+1)*max(0, users_infos(2, sub6_users) - achieved_means(sub6_users)) + users_infos(2, sub6_users);
            users_infos_mmW(2,:) = (i+1)*max(0, users_infos(2, mmW_users) - achieved_means(mmW_users)) + users_infos(2, mmW_users);
            %users_infos_sub6(2,:) = (i+1)*(users_infos(2, sub6_users) - achieved_means(sub6_users)) + achieved_means(sub6_users)
            %users_infos_mmW(2,:) = (i+1)*(users_infos(2, mmW_users) - achieved_means(mmW_users)) + achieved_means(mmW_users)
            
        end
        achieved = achieved_means;
        prob_rates = successfulRate_epoch./max_epoch;
        prob_delay = failDelay_epoch./max_epoch;
        average_delay = delay_means;
        time = toc;
        
    end


    %Function to compute the capacity achieved by a given schedule
    function capacity = F(schedule, SINR, band, scheduling_frameTime, N, K)
        ratio_slot_used = ratio_slot(schedule, scheduling_frameTime, N, K);
        capacity_mat =  bsxfun(@times, permute(schedule,[1 3 2]), log(1+SINR));
        capacity = sumcol(capacity_mat);
        capacity = (capacity * band) .* ratio_slot_used;
        capacity = sumall(capacity);
        %capacity = sum(capacity_mat, 'all');
    end
    
    
    %Function to compute the dispersion achieved by a given schedule
    function dispersion = V(schedule, SINR, Q_inv, l, band, scheduling_frameTime, N, K)
        ratio_slot_used = ratio_slot(schedule, scheduling_frameTime, N, K);
        disp = bsxfun(@times, permute(schedule,[1 3 2]), log(exp(1))^2 * (1 - 1./((1+SINR).^2)));
        disp_k = (Q_inv./sqrt(l))'.*real(sqrt(sumcol(disp)));
        disp_k = (disp_k * band) .* ratio_slot_used;
        dispersion = real(sumall(disp_k));
    end
    
    
    %Function to compute the W value (sum of all entries of a given schedule)
    function summ = W(schedule)
        summ = sumall(schedule);
    end
    
    %Function to compute the E value (sum of all entries squared of a given
    %schedule)
    function summ_sq = E(schedule)
        summ_sq = sumall(schedule.^2);
    end
    
    %Function to compute the original objective function
    function u_tot = U(schedule, SINR, Q_inv, l, beta, band)
        u_tot = -F(schedule, SINR, band) + V(schedule, SINR, Q_inv, l, band) + beta*(W(schedule) - E(schedule));
    end
    
    %Function that compute the gradient of the E function (the
    %multiplication with (x - x^(j)) is already included
    function summ_gradE = gradE(schedule, schedule_init)
        double_sched = 2*schedule_init;
        diff = schedule - schedule_init;
        summ_gradE = sumall(double_sched.*diff);
    end
    
    %Helper function to invert a given matrix and avoid 0-division
    function invers = inv_ifNotZero(value)
        mask = value > 0;
        value(mask) = 1./(value(mask));
        value(~mask) = 0;
        invers = value;
    end
    
    %Function that compute the gradient of the V function (the
    %multiplication with (x - x^(j)) is already included
    function summ_gradV = gradV(schedule, schedule_init, SINR, Q_inv, l, band, schedule_frameTime, N, K)
        ratio_slot_used = ratio_slot(schedule, schedule_frameTime, N, K);
        disp = bsxfun(@times, permute(schedule_init,[1 3 2]), log(exp(1))^2 * (1 - 1./((1+SINR).^2)));
        grad_leftPart = (Q_inv./sqrt(l))'.*inv_ifNotZero( real(sqrt(sumcol(disp))) );
        grad_rightPart = 0.5*log(exp(1))^2*(1 - 1./((1+SINR).^2));
        grad_mat = bsxfun(@times, grad_leftPart, grad_rightPart);
        diff = schedule - schedule_init;
        gradV_full = bsxfun(@times, grad_mat, permute(diff, [1 3 2]));
        gradV_full = (gradV_full * band)  .* ratio_slot_used;
        summ_gradV = sumall(gradV_full);
    end
    
    %Function to compute the modified and convex objective function (after 1st
    %order approx.)
    function obj = U_bar(schedule, schedule_init, SINR, Q_inv, l, beta, band, schedule_frameTime, N, K)
        %obj = -F(schedule, SINR, band, schedule_frameTime, N, K) + V(schedule_init, SINR, Q_inv, l, band, schedule_frameTime, N, K) + gradV(schedule, schedule_init, SINR, Q_inv, l, band, schedule_frameTime, N, K) + beta*(W(schedule) - E(schedule_init) - gradE(schedule, schedule_init));
        obj = -F_obj(schedule, SINR) + V_obj(schedule_init, SINR, Q_inv, l) + gradV_obj(schedule, schedule_init, SINR, Q_inv, l) + beta*(W(schedule) - E(schedule_init) - gradE(schedule, schedule_init));
    end
    
    
    %Helper function to compute the Capacity (F value) of each user and stored in a size
    %K_sub6 array
    function capacity_mat = F_k(schedule, SINR, band, scheduling_frameTime, N, K)
        ratio_slot_used = ratio_slot(schedule, scheduling_frameTime, N, K);
        capacities =  bsxfun(@times, permute(schedule,[1 3 2]), log(1+SINR));
        capacity_mat = sumcol(capacities);
        capacity_mat = (capacity_mat * band) .* ratio_slot_used;
    end
    
    %Helper function to compute the Dispersion (V value) of each user and stored in a size
    %K_sub6 array
    function dispersion_mat = V_k(schedule, SINR, Q_inv, l, band, scheduling_frameTime, N, K)
        ratio_slot_used = ratio_slot(schedule, scheduling_frameTime, N, K);
        disp = bsxfun(@times, permute(schedule,[1 3 2]), log(exp(1))^2 * (1 - 1./((1+SINR).^2)));
        dispersion_mat = (Q_inv./sqrt(l))'.*real(sqrt(sumcol(disp)));
        dispersion_mat = (dispersion_mat * band) .* ratio_slot_used;
    end
    
    function gradV_mat = gradV_k(schedule, schedule_init, SINR, Q_inv, l, band, schedule_frameTime, N, K)
        ratio_slot_used = ratio_slot(schedule, schedule_frameTime, N, K);
        disp = bsxfun(@times, permute(schedule_init,[1 3 2]), log(exp(1))^2 * (1 - 1./((1+SINR).^2)));
        grad_leftPart = (Q_inv./sqrt(l))'.*inv_ifNotZero( real(sqrt(sumcol(disp))));
        grad_rightPart = 0.5*log(exp(1))^2*(1 - 1./((1+SINR).^2));
        grad_mat = bsxfun(@times, grad_leftPart, grad_rightPart);
        diff = schedule - schedule_init;
        gradV_full = bsxfun(@times, grad_mat, permute(diff, [1 3 2]));
        gradV_full = (gradV_full * band) .* ratio_slot_used;
        gradV_mat = sumcol(gradV_full);
    end
    
    %constraint creation into function, c: c(x) <= 0, ceq: ceq(x) = 0
    function [c,ceq] = allcons(sched, schedule_init, SINR, Q_inv, l, user_infos, K, N, band, schedule_frameTime)
        c = [];
    
        c_1 = W(sched) - E(sched);
        c_2 = sum(sched, 3) - 1;
        %c_3 = [];
        c_3 = sumcol(sched) - N;
        c = [c c_1(:)' c_2(:)' c_3(:)'];
    
        ceq = [];
    
        capacity_mat = F_k(sched, SINR, band, schedule_frameTime, N, K);
        disp_mat = V_k(schedule_init, SINR, Q_inv, l, band, schedule_frameTime, N, K);
        gradV_mat = gradV_k(sched, schedule_init, SINR, Q_inv, l, band, schedule_frameTime, N, K);
    
        %delay = calcul_delay(sched, K, schedule_frameTime);
        %c_delay = delay - user_infos(4,:);
    
        for k = 1:K
        
            delay_k = user_infos(4,k);
            cons_delay_k = sched(:,delay_k+1:end,k); %Toute la colonne, ligne k+1 jusqu'à la fin, user k -> pour tous les RB, frametime : delai_K +1 à fin, tt user : On veut que les délais trop grands soit = 0
            ceq = [ceq cons_delay_k];
        
            rate_k = user_infos(2,k);
            cons_rate_k = - capacity_mat(k) + disp_mat(k) + gradV_mat(k) + rate_k;
            c = [c cons_rate_k];
        end
    
    end
    
    function q_inv = getQ_inv(error)
        q_inv = sqrt(2)*erfinv(1 - 2*error);
    end
    
    function SINR = getGlobalSINR_sub6
        %global SINR_sub6
        SINR = SINR_sub6;
    end
    
    function infos = getGlobalUser_infos
        %global user_infos
        infos = user_infos;
    end
    
    function Qi = getGlobalQinv
        %global Q_inv
        Qi = Q_inv;
    end
    
    function init = getGlobalSched_init
        %global schedule_init
        init = schedule_init;
    end
    
    function L = getGlobalL
        %global l
        L = l;
    end
    
    function K = getGlobalK_sub6
        %global K_sub6
        K = K_sub6;
    end
    
    function N = getGlobalN_sub6
        %global N_sub6
        N = N_sub6;
    end
    
    function K = getGlobalK_mmW
        %global K_mmW
        K = K_mmW;
    end
    
    function N = getGlobalN_mmW
        %global N_mmW
        N = N_mmW;
    end
    
    function FT = getGlobalschedule_frametime
        %global schedule_frameTime
        FT = schedule_frameTime;
    end
    
    function bta = getGlobalBeta
        %global beta
        bta = beta;
    end
    
    
    function delay = calcul_delay(schedule, K, schedule_frameTime)
        delay = zeros(1,K);
        for k= 1:K
            for x = 1:schedule_frameTime
            
                if sum(schedule(:,x,k)) >= 1
                    delay(k) = x;
                    break;
                end
            end
        end
    end
    
    function slot_used_k = ratio_slot(sched, scheduling_frameTime, N, K)
        ratio_slot_used_k = zeros(K, N);
        slot_used_k = zeros(1,K);
        for k = 1:K
            for n = 1:N
                ratio_slot_used_k(k,n) = sum(sched(n, :, k)) / scheduling_frameTime;
            end
            slot_used_k(k) = sum(ratio_slot_used_k(k,:)) / N;
        end
    end
    
    function sched_init = init_schedule(N, K, schedule_frameTime) %% Schedule init v1
        sched_init = zeros(N,schedule_frameTime, K);
        time = 1;
        for k = 1:K
            sched_init(:,time,k) = 1;
            time = time + 1;
            if time > schedule_frameTime
                break
            end
        end
    end
    
    function sched_init = init_schedule2(N, K, schedule_frameTime)  %% Schedule init v2
        sched_init = zeros(N, schedule_frameTime, K);
        cpt_k = 1;
        t = 1;
        cpt_n = 1;
        for i = 1: N * schedule_frameTime
            sched_init(cpt_n,t,cpt_k) = 1;
            if mod(t, schedule_frameTime) == 0
                t = 1;
                cpt_n = cpt_n + 1;
            else
                t = t + 1;
            end
            if mod(cpt_k, K) == 0
                cpt_k = 1;
            else
                cpt_k = cpt_k + 1;
            end
        end
    end
    
    %Function to compute the dispersion achieved by a given schedule
    function dispersion = V_obj(schedule, SINR, Q_inv, l)
        disp = bsxfun(@times, permute(schedule,[1 3 2]), log(exp(1))^2 * (1 - 1./((1+SINR).^2)));
        disp_k = (Q_inv./sqrt(l))'.*real(sqrt(sumcol(disp)));
        dispersion = real(sumall(disp_k));
    end
    
    %Function to compute the capacity achieved by a given schedule
    function capacity = F_obj(schedule, SINR)
        capacity_mat =  bsxfun(@times, permute(schedule,[1 3 2]), log(1+SINR));
        capacity = sumcol(capacity_mat);
        capacity = sumall(capacity);
    end
    
    %Function that compute the gradient of the V function (the
    %multiplication with (x - x^(j)) is already included
    function summ_gradV = gradV_obj(schedule, schedule_init, SINR, Q_inv, l)
        dispe = bsxfun(@times, permute(schedule_init,[1 3 2]), log(exp(1))^2 * (1 - 1./((1+SINR).^2)));
        grad_leftPart = (Q_inv./sqrt(l))'.*inv_ifNotZero( real(sqrt(sumcol(dispe))) );
        grad_rightPart = 0.5*log(exp(1))^2*(1 - 1./((1+SINR).^2));
        grad_mat = bsxfun(@times, grad_leftPart, grad_rightPart);
        diff = schedule - schedule_init;
        gradV_full = bsxfun(@times, grad_mat, permute(diff, [1 3 2]));
        summ_gradV = sumall(gradV_full);
    end

    
    %%% Shannon 
    function objective = U_Shannon(schedule, SINR)
        objective = - F_obj(schedule, SINR) + beta*(W(schedule) - E(schedule));
    end

    %constraint creation into function, c: c(x) <= 0, ceq: ceq(x) = 0
    function [c,ceq] = allcons_Shannon(sched, SINR, user_infos, K, N, band, schedule_frameTime)
        c = [];
    
        c_1 = W(sched) - E(sched);
        c_2 = sum(sched, 3) - 1;
        %c_3 = [];
        c_3 = sumcol(sched) - N;
        c = [c c_1(:)' c_2(:)' c_3(:)'];
    
        ceq = [];
    
        capacity_mat = F_k(sched, SINR, band, schedule_frameTime, N, K);
    
        %delay = calcul_delay(sched, K, schedule_frameTime);
        %c_delay = delay - user_infos(4,:);
    
        for k = 1:K
        
            delay_k = user_infos(4,k);
            cons_delay_k = sched(:,delay_k+1:end,k); %Toute la colonne, ligne k+1 jusqu'à la fin, user k -> pour tous les RB, frametime : delai_K +1 à fin, tt user : On veut que les délais trop grands soit = 0
            ceq = [ceq cons_delay_k];
        
            rate_k = user_infos(2,k);
            cons_rate_k = - capacity_mat(k) + rate_k;
            c = [c cons_rate_k];
        end
    
    end



    function summ = sumall(sched)
        summ = 0;
        sz = size(sched);
        N = sz(1);
        frameTime = sz(2);
        if length(sz) == 3
            K = sz(3);
        else
            K = 1;
        end
        
        for n = 1:N
            for t = 1:frameTime
                for k = 1:K
                    summ = summ + sched(n,t,k);
                end
            end
        end
    end
        
    function summ = sumcol(sched)
        sz = size(sched);
        N = sz(1);
        frameTime = sz(2);
        if length(sz) == 3
            K = sz(3);
        else
            K = 1;
        end
        summ = zeros(1, frameTime);
        for t = 1:frameTime
            for n = 1:N
                for k = 1:K
                    summ(t) = summ(t) + sched(n,t,k);
                end
            end
        end
    end

end