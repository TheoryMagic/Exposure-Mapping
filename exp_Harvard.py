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
                                eta=0.028,
                                seed=123):
    np.random.seed(seed)

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
                'Method': design_name,
                'ATE_est': ate_seq[i],
                'V_i': V_is[i],
                'upper': upper_seq[i],
                'lower': lower_seq[i],
                'covered': int( 
                    (true_mu1 - true_mu0) >= lower_seq[i] and 
                    (true_mu1 - true_mu0) <= upper_seq[i] 
                ),
                'contain_zero': int( 0 >= lower_seq[i] and 0 <= upper_seq[i] ),
                'reward': rewards_seq[i],
                'instant_regret': instant_regret[i],
                'cumsum_regret': cumsum_regret[i]
            }
            records.append(rec)

    return pd.DataFrame(records)


##############################################################################
# 5) 主函数：对比 Bernoulli / Standard / Unclipped MAD 并绘制4张图
##############################################################################
if __name__ == '__main__':
    N = 100
    t_steps = 10000
    alpha = 0.05
    true_mu0 = 0.6
    true_mu1 = 0.8
    true_ATE = true_mu1 - true_mu0

    eta = np.sqrt(
        -2*np.log(alpha) + np.log(-2*np.log(alpha)+1)
    ) / np.sqrt(t_steps)
    print("Computed eta =", eta)

    def delta_unclipped(t):
        return 1.0/(t**0.24)

    # 跑三个实验
    df_bern = run_experiments_and_collect(
        N, t_steps, alpha,
        true_mu0, true_mu1,
        design_name='Bernoulli',
        eta=eta,
        seed=111
    )
    df_std = run_experiments_and_collect(
        N, t_steps, alpha,
        true_mu0, true_mu1,
        design_name='Standard',
        eta=eta,
        seed=222
    )
    df_unclip = run_experiments_and_collect(
        N, t_steps, alpha,
        true_mu0, true_mu1,
        delta_func=delta_unclipped,
        design_name='Unclipped MAD',
        eta=eta,
        seed=333
    )

    all_df = pd.concat([df_bern, df_std, df_unclip], ignore_index=True)

    # 计算 CI 宽度
    all_df['width'] = all_df['upper'] - all_df['lower']
    all_df['width'] = all_df['width'].clip(upper=1.0)

    # coverage
    coverage_df = (all_df.groupby(['Method','time'])['covered']
                   .mean().reset_index(name='coverage'))

    # proportion stopped
    all_df['stopped'] = 1 - all_df['contain_zero']
    stop_df = (all_df.groupby(['Method','time'])['stopped']
               .mean().reset_index(name='prop_stopped'))

    # cumulative regret
    regret_df = (all_df.groupby(['Method','time'])['cumsum_regret']
                 .mean().reset_index(name='mean_cum_regret'))

    # width
    width_df = (all_df.groupby(['Method','time'])['width']
                .mean().reset_index(name='mean_width'))

    merged = coverage_df.merge(stop_df, on=['Method','time'])
    merged = merged.merge(regret_df, on=['Method','time'])
    merged = merged.merge(width_df, on=['Method','time'])

    # ==== 画4张图 ====
    # 1) Coverage
    plt.figure(figsize=(8,5))
    sns.lineplot(data=merged, x='time', y='coverage', hue='Method')
    plt.axhline(1 - alpha, ls='--', color='gray', alpha=0.6, label='1 - alpha')
    plt.xscale('log')  # 保持对数坐标
    plt.ylim([0,1.05])
    plt.title(f'Coverage (ATE={true_ATE})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Proportion Stopped
    plt.figure(figsize=(8,5))
    sns.lineplot(data=merged, x='time', y='prop_stopped', hue='Method')
    plt.xscale('log')  # 保持对数坐标
    plt.ylim([0,1.05])
    plt.title('Proportion Stopped')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Cumulative Regret - 改为线性x轴
    plt.figure(figsize=(8,5))
    sns.lineplot(data=merged, x='time', y='mean_cum_regret', hue='Method')
    # ---- 这里不再调用 plt.xscale('log') ----
    plt.title('Cumulative Regret (Linear Time Axis)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) Average Width
    plt.figure(figsize=(8,5))
    sns.lineplot(data=merged, x='time', y='mean_width', hue='Method')
    plt.xscale('log')  # 保持对数坐标
    plt.ylim([0,1.05])
    plt.title('Average Width (Capped at 1.0)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Done!")
