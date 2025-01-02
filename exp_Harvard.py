#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# 1) 置信区间半径计算
##############################################################################
def compute_CS_eta_new(eta, s_hats, alpha, t_steps):
    s_cumsum = np.cumsum(s_hats)
    V_i_list = []
    for t in range(1, t_steps + 1):
        St_val = s_cumsum[t-1]
        inside_sqrt = 2 * (St_val * eta**2 + 1) / (t**2 * eta**2)
        log_term = np.log(np.sqrt(St_val*eta**2 + 1) / alpha)
        log_term = max(log_term, 1e-12)
        V_t = np.sqrt(inside_sqrt * log_term)
        V_i_list.append(V_t)
    return np.array(V_i_list)

##############################################################################
# 2) Thompson Sampling: Standard
##############################################################################
def compute_TS_CSs_standard(mus, alpha, t_steps, arm1_params, arm0_params, eta):
    ates_i   = []
    ate_list = []
    arm_path = []
    s_hats   = []
    rewards  = []

    alpha1, beta1 = arm1_params
    alpha0, beta0 = arm0_params

    for i in range(1, t_steps + 1):
        theta1 = np.random.beta(alpha1, beta1)
        theta0 = np.random.beta(alpha0, beta0)
        chosen_arm = 1 if theta1 >= theta0 else 0
        arm_path.append(chosen_arm)

        y_i = np.random.binomial(1, mus[chosen_arm])

        # 估计 p_1, p_0
        samples1 = np.random.beta(alpha1, beta1, size=1000)
        samples0 = np.random.beta(alpha0, beta0, size=1000)
        p_1 = max(np.mean(samples1 >= samples0), 1e-8)
        p_0 = max(1 - p_1, 1e-8)

        # ate_inc + 截断
        if chosen_arm == 1:
            raw_ate_inc = y_i / p_1
        else:
            raw_ate_inc = -y_i / p_0
        ate_inc = np.clip(raw_ate_inc, -1.0, 1.0)
        ates_i.append(ate_inc)

        # sigma^2_i
        if chosen_arm == 1:
            sigma2_hat_i = (y_i**2)/(p_1**2)
        else:
            sigma2_hat_i = (y_i**2)/(p_0**2)
        s_hats.append(sigma2_hat_i)

        rewards.append(y_i)
        ate_list.append(np.mean(ates_i))

        # 更新 Beta
        if chosen_arm == 1:
            alpha1 += y_i
            beta1  += (1 - y_i)
        else:
            alpha0 += y_i
            beta0  += (1 - y_i)

    return {
        'ate': np.array(ate_list),
        'arm_path': np.array(arm_path),
        'sig_hats': np.array(s_hats),
        'rewards': np.array(rewards)
    }


##############################################################################
# 3) MAD
##############################################################################
def compute_TS_CSs_MAD(mus, alpha, t_steps, arm1_params, arm0_params, delta_array, eta, alg='TS'):
    ates_i   = []
    ate_list = []
    arm_path = []
    s_hats   = []
    rewards  = []

    alpha1, beta1 = arm1_params
    alpha0, beta0 = arm0_params

    for i in range(1, t_steps + 1):
        dt = delta_array[i-1]
        if alg == 'TS':
            samples1 = np.random.beta(alpha1, beta1, size=1000)
            samples0 = np.random.beta(alpha0, beta0, size=1000)
            p_1_est  = np.mean(samples1 >= samples0)
        else:
            p_1_est = 0.5

        if np.random.rand() < dt:
            chosen_arm = np.random.randint(2)
        else:
            if alg == 'TS':
                draw1 = np.random.beta(alpha1, beta1)
                draw0 = np.random.beta(alpha0, beta0)
                chosen_arm = 1 if draw1 >= draw0 else 0
            else:
                chosen_arm = 1 if p_1_est > 0.5 else 0

        arm_path.append(chosen_arm)
        y_i = np.random.binomial(1, mus[chosen_arm])

        p_1_MAD = dt * 0.5 + (1 - dt)*p_1_est
        p_1_MAD = max(p_1_MAD, 1e-8)
        p_0_MAD = max(1.0 - p_1_MAD, 1e-8)

        if chosen_arm == 1:
            raw_ate_inc = y_i / p_1_MAD
        else:
            raw_ate_inc = -y_i / p_0_MAD
        ate_inc = np.clip(raw_ate_inc, -1.0, 1.0)
        ates_i.append(ate_inc)

        if chosen_arm == 1:
            sigma2_hat_i = (y_i**2)/(p_1_MAD**2)
        else:
            sigma2_hat_i = (y_i**2)/(p_0_MAD**2)
        s_hats.append(sigma2_hat_i)

        rewards.append(y_i)
        ate_list.append(np.mean(ates_i))

        if alg == 'TS':
            if chosen_arm == 1:
                alpha1 += y_i
                beta1  += (1 - y_i)
            else:
                alpha0 += y_i
                beta0  += (1 - y_i)

    return {
        'ate': np.array(ate_list),
        'arm_path': np.array(arm_path),
        'sig_hats': np.array(s_hats),
        'rewards': np.array(rewards)
    }


##############################################################################
# 4) 运行多次并汇总
##############################################################################
def run_experiments_and_collect(N, t_steps, alpha,
                                true_mu0=0.5, true_mu1=0.8,
                                delta_func=None,
                                design_name='MAD',
                                eta=0.028):

    records = []
    mus = [true_mu0, true_mu1]
    best_mu = max(true_mu0, true_mu1)   # 最优臂的均值

    for rep_idx in range(N):
        arm1_params = [1.0, 1.0]
        arm0_params = [1.0, 1.0]

        if design_name == 'Bernoulli':
            delta_array = np.ones(t_steps)
            results = compute_TS_CSs_MAD(
                mus, alpha, t_steps, arm1_params, arm0_params,
                delta_array, eta, alg='TS'
            )
        elif design_name == 'Standard':
            results = compute_TS_CSs_standard(
                mus, alpha, t_steps, arm1_params, arm0_params, eta
            )
        else:
            if delta_func is None:
                def default_deltafunc(x):
                    return 1.0/(x**0.24)
                delta_func = default_deltafunc
            d_arr = np.array([delta_func(i) for i in range(1, t_steps+1)])
            results = compute_TS_CSs_MAD(
                mus, alpha, t_steps, arm1_params, arm0_params,
                d_arr, eta, alg='TS'
            )

        sig_hats = results['sig_hats']
        V_is = compute_CS_eta_new(eta, sig_hats, alpha, t_steps)
        ate_seq = results['ate']
        upper_seq = ate_seq + V_is
        lower_seq = ate_seq - V_is
        rewards_seq = results['rewards']
        arm_path    = results['arm_path']

        # 后悔(使用 "best_mu - mus[chosen_arm]")
        instant_regret = []
        for i, arm_chosen in enumerate(arm_path):
            mu_chosen = mus[arm_chosen]
            regret_i = best_mu - mu_chosen
            instant_regret.append(regret_i)
        cumsum_regret = np.cumsum(instant_regret)

        for i in range(t_steps):
            rec = {
                'replicate': rep_idx+1,
                'time': i+1,
                'alpha': alpha,
                'Method': design_name,
                'ATE_est': ate_seq[i],
                'V_i': V_is[i],
                'upper': upper_seq[i],
                'lower': lower_seq[i],
                'reward': rewards_seq[i],
                'instant_regret': instant_regret[i],
                'cumsum_regret': cumsum_regret[i]
            }
            records.append(rec)

    return pd.DataFrame(records)


##############################################################################
# 5) 主函数：对比不同 alpha 的方法并绘制 Regret 和 Width
##############################################################################
if __name__ == '__main__':
    N = 100
    t_steps = 10000
    true_mu0 = 0.5
    true_mu1 = 0.8
    alphas = [0.24, 0.2, 0.15, 0.1, 0.05]
    methods = ['Bernoulli', 'Standard', 'Unclipped MAD']
    colors = {'Bernoulli': 'blue', 'Standard': 'red', 'Unclipped MAD': 'green'}
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # 预定义五种线型 
    all_dfs = []

    for alpha in alphas:
        eta = np.sqrt(
            -2*np.log(alpha) + np.log(-2*np.log(alpha)+1)
        ) / np.sqrt(t_steps)

        for method in methods:
            if method == 'Unclipped MAD':
                delta_func = lambda t: 1.0/(t**0.24)
            else:
                delta_func = None

            np.random.seed(1337)

            df = run_experiments_and_collect(
                N, t_steps, alpha,
                true_mu0, true_mu1,
                delta_func=delta_func,
                design_name=method,
                eta=eta
            )
            all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)

    # 计算 CI 宽度
    all_df['width'] = all_df['upper'] - all_df['lower']
    all_df['width'] = all_df['width'].clip(upper=1.0)

    # cumulative regret
    regret_df = (all_df.groupby(['alpha', 'Method', 'time'])
                 .agg(mean_cum_regret=('cumsum_regret', 'mean'),
                      se_cum_regret=('cumsum_regret', 'sem'))
                 .reset_index())

    # average width
    width_df = (all_df.groupby(['alpha', 'Method', 'time'])
                .agg(mean_width=('width', 'mean'),
                     se_width=('width', 'sem'))
                .reset_index())

    # ==== 画图 ====
    # 1) Cumulative Regret (线性 x 轴)
    plt.figure(figsize=(10,6))
    for method in methods:
        for idx, alpha in enumerate(alphas):
            sub_df = regret_df[(regret_df['Method'] == method) & (regret_df['alpha'] == alpha)]
            plt.plot(sub_df['time'], sub_df['mean_cum_regret'], label=f'{method} (alpha={alpha})',
                     color=colors[method], linestyle=styles[idx])
            plt.fill_between(sub_df['time'],
                             sub_df['mean_cum_regret'] - 2 * sub_df['se_cum_regret'],
                             sub_df['mean_cum_regret'] + 2 * sub_df['se_cum_regret'],
                             alpha=0.2, color=colors[method])
    plt.title('Cumulative Regret Across Methods and Alphas')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.legend(title='Method (Alpha)')
    plt.tight_layout()
    plt.show()

    # 2) Average Width (对数 x 轴)
    plt.figure(figsize=(10,6))
    for method in methods:
        for idx, alpha in enumerate(alphas):
            sub_df = width_df[(width_df['Method'] == method) & (width_df['alpha'] == alpha)]
            plt.plot(sub_df['time'], sub_df['mean_width'], label=f'{method} (alpha={alpha})',
                     color=colors[method], linestyle=styles[idx])
            plt.fill_between(sub_df['time'],
                             sub_df['mean_width'] - 2 * sub_df['se_width'],
                             sub_df['mean_width'] + 2 * sub_df['se_width'],
                             alpha=0.2, color=colors[method])
    plt.xscale('log')
    plt.title('Average Width Across Methods and Alphas')
    plt.xlabel('Time')
    plt.ylabel('Average Width')
    plt.legend(title='Method (Alpha)')
    plt.tight_layout()
    plt.show()

    print("Done!")
