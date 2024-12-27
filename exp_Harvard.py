#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
示例：复现 Mixture Adaptive Design (MAD) 与 Bernoulli、Standard (TS) 等设计的对比实验（Bernoulli Rewards）。
请先 `pip install numpy pandas seaborn matplotlib`。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1) 置信区间半径计算函数 (根据论文给出的 anytime-valid CS 公式)
def compute_CS_eta(eta, s_hats, alpha, t_steps):
    """
    s_hats: 长度 t_steps，每一步的 sigma^2 估计
    返回: V_i 数组，长度 t_steps
    """
    s_cumsum = np.cumsum(s_hats)
    i_arr = np.arange(1, t_steps + 1)
    V_i_list = []

    for i, scum in zip(i_arr, s_cumsum):
        # inside = (sum_{k=1 to i} s_k * eta^2 + 1) / eta^2
        inside = (scum * (eta**2) + 1) / (eta**2)
        # V_i = (1/i) * sqrt( inside * log( inside / alpha^2 ) )
        if inside <= 0 or (inside / (alpha**2)) <= 1e-12:
            # 防止对数出错
            V_i_list.append(0.0)
        else:
            val = (1.0 / i) * np.sqrt( inside * np.log( inside / (alpha**2) ) )
            V_i_list.append(val)

    return np.array(V_i_list)


# 2) 纯 Thompson Sampling (Beta-Bernoulli) 设计
def compute_TS_CSs_standard(mus, alpha, t_steps, arm1_params, arm0_params, eta):
    """
    mus: [p0, p1], Bernoulli 成功率
    arm1_params, arm0_params: [alpha, beta] 形参 (初始都= [1,1])
    返回包含 ate、arm_path、sig_hats、rewards
    """
    ates_i = []
    ate_list = []
    arm_path = []
    s_hats = []
    rewards = []

    alpha1, beta1 = arm1_params
    alpha0, beta0 = arm0_params

    for i in range(1, t_steps + 1):
        # Thompson Sampling: 从各自 Beta 分布采样
        theta1 = np.random.beta(alpha1, beta1)
        theta0 = np.random.beta(alpha0, beta0)
        chosen_arm = 1 if theta1 >= theta0 else 0
        arm_path.append(chosen_arm)

        # 产生伯努利收益
        y_i = np.random.binomial(1, mus[chosen_arm])

        # 估计 p_1(= P(arm1>arm0))，这里用蒙特卡洛近似
        samples1 = np.random.beta(alpha1, beta1, size=1000)
        samples0 = np.random.beta(alpha0, beta0, size=1000)
        p_1 = np.mean(samples1 >= samples0)
        p_1 = max(p_1, 1e-8)  # 防止除 0
        p_0 = 1.0 - p_1
        p_0 = max(p_0, 1e-8)

        # 根据 chosen_arm 计算 ATE increment
        if chosen_arm == 1:
            ate_inc = y_i / p_1
            sigma2_hat_i = (y_i**2) / (p_1**2)
        else:
            ate_inc = -y_i / p_0
            sigma2_hat_i = (y_i**2) / (p_0**2)

        ates_i.append(ate_inc)
        s_hats.append(sigma2_hat_i)
        rewards.append(y_i)

        current_ate = np.mean(ates_i)
        ate_list.append(current_ate)

        # 更新 Beta(α, β)
        if chosen_arm == 1:
            alpha1 += y_i
            beta1 += (1 - y_i)
        else:
            alpha0 += y_i
            beta0 += (1 - y_i)

    return {
        'ate': np.array(ate_list),
        'arm_path': np.array(arm_path),
        'sig_hats': np.array(s_hats),
        'rewards': np.array(rewards)
    }


# 3) Mixture Adaptive Design (MAD):
#    以概率 delta_t 随机选臂(0.5-0.5)，以概率 (1-delta_t) 用 TS 选臂
def compute_TS_CSs_MAD(mus, alpha, t_steps, arm1_params, arm0_params, delta_array, eta, alg='TS'):
    """
    delta_array: 长度 t_steps 的向量, delta_t = f(t)
    alg: 默认'TS' (也可扩展为 'UCB' 等)
    返回 dict 同上
    """
    ates_i = []
    ate_list = []
    arm_path = []
    s_hats = []
    rewards = []

    alpha1, beta1 = arm1_params
    alpha0, beta0 = arm0_params

    for i in range(1, t_steps + 1):
        dt = delta_array[i-1]
        # 估计 p_1(=P(arm1>arm0))，仅当 alg=TS 时用 Beta 抽样
        if alg == 'TS':
            samples1 = np.random.beta(alpha1, beta1, size=1000)
            samples0 = np.random.beta(alpha0, beta0, size=1000)
            p_1_est = np.mean(samples1 >= samples0)
        else:
            p_1_est = 0.5  # 仅示例

        # 先以概率 dt 均匀随机选臂
        if np.random.rand() < dt:
            chosen_arm = np.random.randint(2)
        else:
            # 否则用 TS 选
            if alg == 'TS':
                draw1 = np.random.beta(alpha1, beta1)
                draw0 = np.random.beta(alpha0, beta0)
                chosen_arm = 1 if draw1 >= draw0 else 0
            else:
                chosen_arm = 1 if p_1_est > 0.5 else 0

        arm_path.append(chosen_arm)

        # 产生伯努利收益
        y_i = np.random.binomial(1, mus[chosen_arm])

        # p_1_MAD = dt*0.5 + (1-dt)* p_1_est
        p_1_MAD = dt * 0.5 + (1 - dt) * p_1_est
        p_1_MAD = max(p_1_MAD, 1e-8)
        p_0_MAD = 1.0 - p_1_MAD
        p_0_MAD = max(p_0_MAD, 1e-8)

        if chosen_arm == 1:
            ate_inc = y_i / p_1_MAD
            sigma2_hat_i = (y_i**2) / (p_1_MAD**2)
        else:
            ate_inc = -y_i / p_0_MAD
            sigma2_hat_i = (y_i**2) / (p_0_MAD**2)

        ates_i.append(ate_inc)
        s_hats.append(sigma2_hat_i)
        rewards.append(y_i)

        current_ate = np.mean(ates_i)
        ate_list.append(current_ate)

        # 更新 Beta(α, β)
        if alg == 'TS':
            if chosen_arm == 1:
                alpha1 += y_i
                beta1 += (1 - y_i)
            else:
                alpha0 += y_i
                beta0 += (1 - y_i)
        else:
            pass

    return {
        'ate': np.array(ate_list),
        'arm_path': np.array(arm_path),
        'sig_hats': np.array(s_hats),
        'rewards': np.array(rewards)
    }


# 4) 针对某个 design，多次实验并存储结果
def run_experiments_and_collect(N, t_steps, alpha,
                                true_mu0=0.5, true_mu1=0.8,
                                delta_func=None,
                                clipped=False,
                                design_name='MAD',
                                eta=0.028,
                                seed=123):
    """
    N: 重复次数
    t_steps: 总时间步
    alpha: 显著性水平 (0.05)
    true_mu0, true_mu1: 两个臂的真实伯努利成功率
    delta_func: 用于生成 delta_t 的函数，例如 lambda t: 1/t^0.24
    clipped: 若 True 则 delta_t = max(delta_t, 0.2)
    design_name: 'Bernoulli', 'Standard', 'MAD'(unclipped) or 'MAD Clipped'等
    eta: 计算CS时用的参数
    seed: 随机种子
    
    返回：DataFrame，每行包含 replicate, time, Method, ATE_est, V_i, upper, lower, covered, contain_zero, reward
    """
    np.random.seed(seed)

    records = []
    true_ATE = true_mu1 - true_mu0
    mus = [true_mu0, true_mu1]

    for rep_idx in range(N):
        # 每次重复都要重置 Beta(1,1)
        arm1_params = [1.0, 1.0]
        arm0_params = [1.0, 1.0]

        if design_name == 'Bernoulli':
            # 纯 Bernoulli: delta(t)=1 => 每步都 0.5-0.5 随机
            delta_array = np.ones(t_steps)
            # 这里直接套用 MAD 函数，但 delta=1
            results = compute_TS_CSs_MAD(mus, alpha, t_steps, arm1_params[:], arm0_params[:],
                                         delta_array, eta, alg='TS')
        elif design_name == 'Standard':
            # 纯 TS: delta(t)=0
            results = compute_TS_CSs_standard(mus, alpha, t_steps,
                                              arm1_params[:], arm0_params[:], eta)
        else:
            # MAD (默认) => delta_func(t)
            if delta_func is None:
                # 若没传入，就给一个默认
                def delta_func_default(x):
                    return 1.0 / (x**0.24)
                delta_func = delta_func_default

            d_arr = np.array([delta_func(i) for i in range(1, t_steps+1)])
            if clipped:
                d_arr = np.maximum(d_arr, 0.2)
            
            results = compute_TS_CSs_MAD(mus, alpha, t_steps,
                                         arm1_params[:], arm0_params[:],
                                         d_arr, eta, alg='TS')

        # 计算置信区间半径 V_i
        sig_hats = results['sig_hats']
        V_is = compute_CS_eta(eta, sig_hats, alpha, t_steps)
        ate_seq = results['ate']
        
        upper_seq = ate_seq + V_is
        lower_seq = ate_seq - V_is
        rewards_seq = results['rewards']

        for i in range(t_steps):
            covered = 1 if (true_ATE >= lower_seq[i] and true_ATE <= upper_seq[i]) else 0
            contain_zero = 1 if (0 >= lower_seq[i] and 0 <= upper_seq[i]) else 0
            records.append({
                'replicate': rep_idx+1,
                'time': i+1,
                'Method': design_name,
                'ATE_est': ate_seq[i],
                'V_i': V_is[i],
                'upper': upper_seq[i],
                'lower': lower_seq[i],
                'covered': covered,
                'contain_zero': contain_zero,
                'reward': rewards_seq[i],
            })

    df = pd.DataFrame(records)
    return df


if __name__ == '__main__':

    # ============== 参数区 ==============
    N = 100            # 重复实验次数
    t_steps = 10000    # 总时间步
    alpha = 0.05       # 显著性水平
    true_mu0 = 0.5
    true_mu1 = 0.8
    true_ATE = true_mu1 - true_mu0

    # 来自 Waubdy-Smith 2023 推荐
    eta = np.sqrt(-2*np.log(alpha) + np.log(-2*np.log(alpha)+1)) / np.sqrt(t_steps)
    print("Computed eta =", eta)  # 大约 0.028 左右

    # 定义一个用于 unclipped MAD 的 delta(t) 函数
    def delta_unclipped(t):
        return 1.0 / (t**0.24)

    # ============ 跑四种设计，生成结果 =============
    # 1) Bernoulli Design
    df_bern = run_experiments_and_collect(
        N, t_steps, alpha,
        true_mu0=true_mu0, true_mu1=true_mu1,
        delta_func=None,   # 不需要
        clipped=False,
        design_name='Bernoulli',
        eta=eta,
        seed=111
    )
    # 2) Standard Bandit (纯 TS)
    df_std = run_experiments_and_collect(
        N, t_steps, alpha,
        true_mu0=true_mu0, true_mu1=true_mu1,
        design_name='Standard',
        eta=eta,
        seed=222
    )
    # 3) Unclipped MAD
    df_unclip = run_experiments_and_collect(
        N, t_steps, alpha,
        true_mu0=true_mu0, true_mu1=true_mu1,
        delta_func=delta_unclipped,
        clipped=False,
        design_name='Unclipped MAD',
        eta=eta,
        seed=333
    )
    # 4) Clipped MAD
    df_clip = run_experiments_and_collect(
        N, t_steps, alpha,
        true_mu0=true_mu0, true_mu1=true_mu1,
        delta_func=delta_unclipped,
        clipped=True,
        design_name='Clipped MAD',
        eta=eta,
        seed=444
    )

    all_df = pd.concat([df_bern, df_std, df_unclip, df_clip], ignore_index=True)

    # ============ 计算 Coverage / Width / Proportion Stopped / Time-Avg Reward ============

    # CI宽度
    all_df['width'] = all_df['upper'] - all_df['lower']

    # coverage(t) = mean( covered_{rep,t} )
    coverage_df = (all_df
                   .groupby(['Method','time'])['covered']
                   .mean()
                   .reset_index(name='coverage'))

    # proportion stopped: 
    #   定义 stopped=1 当 0 不在CI => contain_zero=0 => stopped=1
    all_df['stopped'] = 1 - all_df['contain_zero']
    stop_df = (all_df
               .groupby(['Method','time'])['stopped']
               .mean()
               .reset_index(name='prop_stopped'))

    # time-avg reward
    all_df = all_df.sort_values(['Method','replicate','time'])
    all_df['cumsum_reward'] = all_df.groupby(['Method','replicate'])['reward'].cumsum()
    all_df['time_avg_reward'] = all_df['cumsum_reward'] / all_df['time']
    reward_df = (all_df
                 .groupby(['Method','time'])['time_avg_reward']
                 .mean()
                 .reset_index(name='mean_reward'))

    # width
    width_df = (all_df
                .groupby(['Method','time'])['width']
                .mean()
                .reset_index(name='mean_width'))

    # 把这些汇总一下
    merged = coverage_df.merge(stop_df, on=['Method','time'])
    merged = merged.merge(reward_df, on=['Method','time'])
    merged = merged.merge(width_df, on=['Method','time'])

    # ============ 画图示例 ============

    # Coverage over time
    plt.figure(figsize=(9,6))
    sns.lineplot(data=merged, x='time', y='coverage', hue='Method')
    plt.axhline(1-alpha, ls='--', color='gray', alpha=0.6, label='1 - alpha')
    plt.xscale('log')
    plt.ylim([0,1.05])
    plt.title(f'Coverage over Time (ATE={true_ATE})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Proportion Stopped
    plt.figure(figsize=(9,6))
    sns.lineplot(data=merged, x='time', y='prop_stopped', hue='Method')
    plt.xscale('log')
    plt.ylim([0,1.05])
    plt.title('Proportion Stopped (Excluding 0) over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Time-Avg Reward
    plt.figure(figsize=(9,6))
    sns.lineplot(data=merged, x='time', y='mean_reward', hue='Method')
    plt.xscale('log')
    plt.title('Time-Averaged Reward')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Width
    plt.figure(figsize=(9,6))
    sns.lineplot(data=merged, x='time', y='mean_width', hue='Method')
    plt.xscale('log')
    plt.title('Average Width of CS')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============ 保存数据 (可选) ============
    merged.to_csv('MAD_experiment_results.csv', index=False)
    print("Done! Results saved to 'MAD_experiment_results.csv'.")
