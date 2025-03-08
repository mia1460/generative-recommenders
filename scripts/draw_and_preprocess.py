import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import csv
from scipy.stats import pearsonr
import re
from matplotlib.ticker import PercentFormatter
import pandas as pd


# delta_x_offsets_0 = torch.tensor([2, 5, 9, 10])
# x = torch.randn(15, 5)
# x = x[delta_x_offsets_0, :]
# print(x)
# cached_v = torch.randn(11, 5)
# print(cached_v)
# v = torch.randn(4, 5)
# print(v)
# v = cached_v.index_copy_(dim=0, index=delta_x_offsets_0, source=v)
# print(v)

def truncate_sequences(row):
    # 将字符串按逗号分割成列表
    item_ids = row['sequence_item_ids'].split(',')
    ratings = row['sequence_ratings'].split(',')
    timestamps = row['sequence_timestamps'].split(',')

    # 检查列表长度是否大于等于200
    if len(item_ids) >= 200:
        # 截取前200个元素
        item_ids = item_ids[:200]
        ratings = ratings[:200]
        timestamps = timestamps[:200]

        # 将处理后的列表重新拼接成字符串
        row['sequence_item_ids'] = ','.join(item_ids)
        row['sequence_ratings'] = ','.join(ratings)
        row['sequence_timestamps'] = ','.join(timestamps)

    return row

def truncate_sequence_data(inputfile, outputfile):
    file_path = inputfile
    df = pd.read_csv(file_path)
    df = df.apply(truncate_sequences, axis=1)

    df.to_csv(outputfile, index=False)
    print(f"data truncated by 200 saved at {outputfile}")

input_file = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_head_1281.csv'
output_file = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_head_1281_max_200.csv'

# truncate_sequence_data(input_file, output_file)


if False:
    # v_source_indices = torch.tensor([0, 1, 2, 5])
    cached_lengths = torch.tensor([2, 3, 4])
    cached_max = torch.sum(cached_lengths)
    print(f"cached_max is {cached_max}")

    cached_v = torch.randn(9, 5)
    print(cached_v)
    cached_v_pad = torch.zeros((14, 5))
    x_offsets = torch.tensor([0, 3, 7])
    v_target_indices = torch.cat([
        torch.arange(x_offsets[i], x_offsets[i]+cached_lengths[i]) for i in range(cached_lengths.shape[0])
    ])
    print(f"v_target_indices is {v_target_indices}")
    cached_v_pad.index_copy_(0, v_target_indices, cached_v)
    print(cached_v_pad)

if False:
    # 创建一个张量，设置requires_grad为True
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    # z = y * y * 3
    # print(z)
    out = y.mean()
    print(out)
    print(x.grad)
    
    # 反向传播计算梯度
    out.backward()
    print(x)
    
    # 输出x的梯度
    print(x.grad)

def parse_log_file(log_file_path, output_csv_path):
    """
    解析日志文件并提取数据保存到 CSV 文件中

    参数:
        log_file_path (str): 日志文件路径
        output_csv_path (str): 输出 CSV 文件路径
    """
    # 读取日志文件内容
    with open(log_file_path, 'r', encoding='utf-8') as file:
        log_content = file.read()

    # 分割日志内容为各个模型块
    model_blocks = re.split(r'(?=\[loading checkpoint\])', log_content)
    model_blocks = [block.strip() for block in model_blocks if block.strip()]

    data = []
    for block in model_blocks:
        # 提取模型名称
        model_match = re.search(r'\[loading checkpoint\] from: .*?/model_(base|\d+)\.ckpt', block)
        model_name = model_match.group(1) if model_match else 'base'

        # print(f"block is {block}\n\n\n")

        # 匹配 `no` 和 `fully` 缓存评估块
        cache_pattern = re.compile(
            r'============== begin evaling use (no|fully) cache\.\.\.==============\n'
            # r'eval use \w+ cache need avg : ([\d.]+) ms\n'
            r'eval use \w+ cache need: ([\d.]+) ms\n'
            r'metrics are: NDCG@10 ([\d.]+), NDCG@50 ([\d.]+), HR@10 ([\d.]+), HR@50 ([\d.]+), MRR ([\d.]+)',
            re.MULTILINE
        )
        cache_matches = cache_pattern.findall(block)

        for match in cache_matches:
            cache_type, time_used, ndcg10, ndcg50, hr10, hr50, mrr = match
            data.append({
                "model": model_name,
                "cache_type": cache_type,
                "time_used": float(time_used),
                "NDCG@10": float(ndcg10),
                "NDCG@50": float(ndcg50),
                "HR@10": float(hr10),
                "HR@50": float(hr50),
                "MRR": float(mrr)
            })

        # 处理 `selective` 缓存评估块，注意额外的一行 `!!!recompute_ratio is XX!!!`
        selective_pattern = re.compile(
            r'============== begin evaling use selective cache\.\.\.==============\n'
            r'!!!recompute_ratio is \d+!!!\n'  # 允许匹配 `recompute_ratio` 这行
            # r'eval use selective cache need avg : ([\d.]+) ms\n'
            r'eval use selective cache need: ([\d.]+) ms\n'
            r'metrics are: NDCG@10 ([\d.]+), NDCG@50 ([\d.]+), HR@10 ([\d.]+), HR@50 ([\d.]+), MRR ([\d.]+)',
            re.MULTILINE
        )
        selective_matches = selective_pattern.findall(block)

        for match in selective_matches:
            time_used, ndcg10, ndcg50, hr10, hr50, mrr = match
            data.append({
                "model": model_name,
                "cache_type": "selective",
                "time_used": float(time_used),
                "NDCG@10": float(ndcg10),
                "NDCG@50": float(ndcg50),
                "HR@10": float(hr10),
                "HR@50": float(hr50),
                "MRR": float(mrr)
            })

    # 转换为 DataFrame 并保存
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"log data saved at: {output_csv_path}")
    return df


# test_3_type_log_path = '/home/yinj@/datas/grkvc/result_logs/test_15_models_3_types.log'
# output_log_csv_path = '/home/yinj@/datas/grkvc/result_logs/test_15_models_3_types.csv'
# df = parse_log_file(test_3_type_log_path, output_log_csv_path)

test_3_type_on_gpu_log_path = '/home/yinj@/datas/grkvc/result_logs/test_15_models_3_types_cache_on_gpu.log'
test_3_type_on_gpu_csv_path = '/home/yinj@/datas/grkvc/result_logs/test_15_models_3_types_cache_on_gpu.csv'
# parse_log_file(test_3_type_on_gpu_log_path, test_3_type_on_gpu_csv_path)

test_3_type_on_gpu_log_avg_time_path = '/home/yinj@/datas/grkvc/result_logs/test_15_models_3_types_cache_on_gpu_w_avg_time.log'
test_3_type_on_gpu_csv_avg_time_path = '/home/yinj@/datas/grkvc/result_logs/test_15_models_3_types_cache_on_gpu_w_avg_time.csv'
# parse_log_file(test_3_type_on_gpu_log_avg_time_path, test_3_type_on_gpu_csv_avg_time_path)


def plot_selected_models(csv_path, selected_models, pic_prefix, output_dir=None):
    """
    可视化指定模型的缓存性能数据
    
    参数:
        csv_path (str): CSV文件路径
        selected_models (list): 需要可视化的模型列表
        output_dir (str): 输出目录 (默认使用预定义路径)
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 设置输出路径
    if not output_dir:
        output_dir = "/home/yinj@/datas/grkvc/graphs"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"selected_models is {selected_models}")

    df["model"] = df["model"].astype(str)
    selected_models = [str(m) for m in selected_models]
    
    # 过滤出选定模型的数据
    filtered_df = df[df["model"].isin(selected_models)]
    # print(filtered_df)
    
    # 配置可视化参数
    cache_colors = {"no": "#4C72B0", "fully": "#DD8452", "selective": "#55A868"}  # 更柔和的配色
    metric_colors = ["#4C72B0", "#DD8452", "#55A868"]
    
    # =====================
    # 绘制运行时间对比图 (2x2 布局)
    # =====================
    fig_time, axes_time = plt.subplots(3, 5, figsize=(10,8))
    fig_time.suptitle("Running Time Comparison by Model", fontsize=12)
    
    for i, model in enumerate(selected_models):
        model_data = filtered_df[filtered_df["model"] == model]
        
        if model_data.empty:
            print(f"Warning: Model {model} not found in data, skipped.")
            continue
        
        # 计算子图位置
        row = i // 5
        col = i % 5
        
        # 绘制运行时间对比
        sns.barplot(
            x="cache_type",
            y="time_used",
            data=model_data,
            palette=cache_colors,
            order=["no", "fully", "selective"],
            width=0.39,
            ax=axes_time[row, col]
        )
        axes_time[row, col].set_title(f"Model {model}", fontsize=14)
        axes_time[row, col].set_xlabel("Cache Type", fontsize=10)
        axes_time[row, col].set_ylabel("Time (s)", fontsize=12)
        axes_time[row, col].set_ylim(0, model_data["time_used"].max() * 1.2)
    
    # 调整布局并保存
    plt.tight_layout()
    time_path = os.path.join(output_dir, pic_prefix+"_running_time_comparison.png")
    plt.savefig(time_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Running time comparison plot saved to: {time_path}")
    
    # =====================
    # 绘制指标对比图 (2x2 布局)
    # =====================
    fig_metrics, axes_metrics = plt.subplots(3, 5, figsize=(15, 12))
    fig_metrics.suptitle("Metrics Comparison by Model", fontsize=26)
    
    for i, model in enumerate(selected_models):
        model_data = filtered_df[filtered_df["model"] == model]
        
        if model_data.empty:
            continue
        
        # 计算子图位置
        row = i // 5
        col = i % 5
        
        # 提取指标数据并转换格式
        metrics_df = model_data.melt(
            id_vars=["cache_type"],
            value_vars=["NDCG@10", "NDCG@50", "HR@10", "HR@50", "MRR"],
            var_name="metric",
            value_name="value"
        )
        
        # 绘制指标对比
        sns.barplot(
            x="metric",
            y="value",
            hue="cache_type",
            data=metrics_df,
            palette=cache_colors,
            hue_order=["no", "fully", "selective"],
            ax=axes_metrics[row, col]
        )
        axes_metrics[row, col].set_title(f"Model {model}", fontsize=24)
        axes_metrics[row, col].set_xlabel("Metric", fontsize=18)
        axes_metrics[row, col].set_ylabel("Value", fontsize=18)
        axes_metrics[row, col].set_ylim(0, 0.8)  # 根据实际数据范围调整
        axes_metrics[row, col].legend(title="Cache Type")
    
    # 调整布局并保存
    plt.tight_layout()
    metric_path = os.path.join(output_dir, pic_prefix+"_metrics_comparison.png")
    plt.savefig(metric_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Metrics comparison plot saved to: {metric_path}")

selected_models = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# plot_selected_models(csv_path=output_log_csv_path, selected_models=selected_models)
# plot_selected_models(csv_path=test_3_type_on_gpu_csv_path, selected_models=selected_models, pic_prefix="test_3_type_on_gpu")
# plot_selected_models(csv_path=test_3_type_on_gpu_csv_avg_time_path, selected_models=selected_models, pic_prefix="test_3_type_on_gpu_csv_avg_time")

def plot_model5_comparison(csv_path, model_num, metric, output_dir=None):
    """
    绘制 Model 5 的运行时间与 HR@50 对比图
    
    参数:
        csv_path (str): CSV文件路径
        output_dir (str): 输出目录 (默认使用预定义路径)
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 设置输出路径
    if not output_dir:
        output_dir = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 筛选 Model 5 的数据
    model5_data = df[df["model"] == model_num]
    
    # 创建画布 (1行2列布局)
    plt.figure(figsize=(12, 6))
    
    # =====================
    # 左图：运行时间对比
    # =====================
    plt.subplot(1, 2, 1)
    sns.barplot(
        x="cache_type",
        y="time_used",
        data=model5_data,
        order=["no", "full", "selective"],
        palette=["#4C72B0", "#DD8452", "#55A868"],  # 学术配色
        saturation=0.8,
        width=0.5,
    )
    plt.title(f"Model {model_num} - Running Time Comparison", fontsize=24)
    plt.xlabel("Cache Strategy", fontsize=22)
    plt.ylabel("Time (seconds)", fontsize=22)
    plt.ylim(0, model5_data["time_used"].max() * 1.2)
    
    # 添加数值标签
    for index, row in model5_data.iterrows():
        plt.text(
            x=index % 3,  # 3种cache_type
            y=row["time_used"] + 0.5,
            s=f"{row['time_used']:.1f}s",
            ha="center",
            fontsize=10
        )
    
    # =====================
    # 右图：HR@50 指标对比
    # =====================
    plt.subplot(1, 2, 2)
    bars = sns.barplot(
        x="cache_type",
        y=metric,
        data=model5_data,
        order=["no", "full", "selective"],
        palette=["#4C72B0", "#DD8452", "#55A868"],
        saturation=0.8,
        width=0.5
    )
    plt.title(f"Model {model_num} - {metric} Comparison", fontsize=24)
    plt.xlabel("Cache Strategy", fontsize=22)
    plt.ylabel("HR@50 Score", fontsize=22)
    plt.ylim(0, 0.7)  # 根据数据范围调整
    
    # 添加数值标签
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            f"{height:.3f}",
            ha="center",
            fontsize=10
        )
    
    # 调整布局并保存
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"model_{model_num}_{metric}_comparison.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"model {model_num} comparison {metric} graph saved at: {output_path}")

log_file_path = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/model_results/step_15s_with_r20.log'
output_csv_path = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/model_results/step_15s_with_r20.csv'
output_img_dir = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/graphs'
graph_name="cache_average_performance_comparison"
log_file_0216_path = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/model_results/step_15s_with_r20_i0_250216.log'
output_csv_0216_path = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/model_results/step_15s_with_r20_i0_250216.csv'
graph_name = ''

selected_models = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# selected_models = [1, 4, 9, 15] 
# selected_models = [5]
# plot_selected_models(csv_path=output_csv_path, selected_models=selected_models)
# plot_selected_models(csv_path=output_csv_0216_path, selected_models=selected_models)

# plot_model5_comparison(output_csv_path, 5, "HR@50")
# plot_model5_comparison(output_csv_0216_path, 8, "HR@50")

# df = parse_log_file(log_file_path=log_file_path, output_csv_path=output_csv_path)
# plot_metrics(df=df, graph_name=graph_name)

# df = parse_log_file(log_file_path=log_file_0216_path, output_csv_path=output_csv_0216_path)
graph_names = {
    'NDCG@10': 'NDCG@10',
    'NDCG@50': 'NDCG@50',
    'HR@10': 'HR@10',
    'HR@50': 'HR@50',
    'MRR': 'MRR'
}
base_dir = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/graphs"
selected_models = [1, 4, 9, 15] 


def plot_kv_diff_cdf(kv_diff_file, graph_name):
    """
    绘制多层 kv_diff 的累积分布函数 (CDF) 图
    
    参数:
        kv_diff_file (str): kv_diff 文件路径
        graph_name (str): 输出图片名称 (不含扩展名)
    """
    # 加载 kv_diff 数据
    kv_diff = torch.load(kv_diff_file)  # List of 8 tensors
    
    # 验证数据结构
    # assert len(kv_diff) == 8, "需要包含 8 个层的 kv_diff 数据"
    kv_diff_len = len(kv_diff)
    
    # 创建画布
    plt.figure(figsize=(10, 6))
    
    # 使用学术配色方案 (Seaborn的深色系)
    # colors = plt.cm.tab10(np.linspace(0, 1, 8))  # 8层不同颜色
    # colors = plt.cm.Blues(np.linspace(0.3, 1, 8))  # 从浅到深的蓝色
    # colors = sns.color_palette("Blues_d", 8)
    # colors = plt.cm.Reds(np.linspace(0.3, 1, 8))
    # colors = plt.cm.Purples(np.linspace(0.3, 1, 8))
    # colors = plt.cm.Oranges(np.linspace(0.3, 1, 8))
    colors = plt.cm.Greys(np.linspace(0.3, 1, kv_diff_len))

    # 定义线型（虚线）
    # linestyles = ['--', '-.', ':', (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (1, 5)), (0, (5, 1)), (0, (3, 1, 1, 1))]
    
    # 遍历每一层
    for layer_idx in range(kv_diff_len):
        # 获取当前层数据并展平
        layer_data = kv_diff[layer_idx].flatten().cpu().numpy()
        
        # 计算 CDF
        sorted_data = np.sort(layer_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制曲线
        plt.plot(sorted_data, 
                cdf, 
                color=colors[layer_idx],
                # linestyle=linestyles[layer_idx],
                linewidth=2,
                alpha=0.8,
                label=f'Layer {layer_idx}')

    # 设置图形格式
    plt.xlabel('kv_diff Value', fontsize=12, labelpad=10)
    plt.ylabel('Cumulative Probability', fontsize=12, labelpad=10)
    plt.title('Cumulative Distribution of kv_diff across Layers', 
             fontsize=14, pad=20)
    
    # 坐标轴增强
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))  # 百分比显示
    plt.xlim(left=0)  # 从0开始
    
    # 图例优化
    plt.legend(loc='lower right', 
              frameon=True, 
              edgecolor='none',
              fontsize=10,
              ncol=2)  # 分两列显示
    
    # 设置背景为白色
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')

    # 保存图片
    base_dir = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/graphs"
    filepath = f"{base_dir}/{graph_name}_cdf.png"
    plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"CDF graph saved at: {filepath}")

kv_diff_path = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/diff_kv/base_step1_diff_kv.pt"
# plot_kv_diff_cdf(kv_diff_file=kv_diff_path, graph_name="base_step1_diff")

def plot_kv_diff_correlation(kv_diff_file, graph_name):
    """
    Plot Pearson correlation of adjacent layer kv_diff within a single model.

    Args:
        kv_diff_file (str): Path to the kv_diff file.
        graph_name (str): Output image save path (e.g., "plot.png").
    """
    # Load kv_diff data
    kv_diff = torch.load(kv_diff_file)  # List of 8 layer tensors

    # Validate structure
    num_layers = len(kv_diff)
    if num_layers < 2:
        raise ValueError("kv_diff must contain at least 2 layers.")

    # Compute Pearson correlations between adjacent layers
    correlations = []
    for i in range(num_layers - 1):
        # Flatten tensors to 1D
        layer_curr = kv_diff[i].flatten().cpu()
        layer_next = kv_diff[i+1].flatten().cpu()

        # Calculate Pearson correlation
        corr, _ = pearsonr(layer_curr.numpy(), layer_next.numpy())
        correlations.append(corr)

    # Plotting
    plt.figure(figsize=(10, 6))
    x_labels = [f"{i}vs{i+1}" for i in range(num_layers-1)]
    bars = plt.bar(x_labels, correlations, color="#1f77b4", alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f"{height:.2f}", ha='center', va='bottom')

    plt.xlabel("Adjacent Layers (i vs i+1)", fontsize=12)
    plt.ylabel("Pearson Correlation", fontsize=12)
    plt.title("Pearson Correlation of Adjacent Layer kv_diff", fontsize=14, pad=20)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    
    # Save and close
    base_dir = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/graphs"
    filepath = os.path.join(base_dir, graph_name + "_pearsonr.png")
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Plot saved to {filepath}")

diff_kv_path = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/diff_kv/base_step1_diff_kv.pt"
# plot_kv_diff_correlation(diff_kv_path, "base_step1_diff")

def remove_last_x_sequences(input_filename, output_filename, begin, end):
    import random
    # 打开输入文件和输出文件
    with open(input_filename, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        # 写入标题行
        writer.writeheader()

        # 逐行读取数据并处理
        for row in reader:
            # print(f"row is {row}")
            # 处理 sequence_item_ids, sequence_ratings, sequence_timestamps
            sequence_item_ids = row['sequence_item_ids'].split(',')
            sequence_ratings = row['sequence_ratings'].split(',')
            sequence_timestamps = row['sequence_timestamps'].split(',')
            x = random.randint(begin, end)
            print(f"x is {x}")
            
            # 确保不去除超过序列长度的部分
            if len(sequence_item_ids) > x:
                sequence_item_ids = sequence_item_ids[:-x]
                sequence_ratings = sequence_ratings[:-x]
                sequence_timestamps = sequence_timestamps[:-x]
            
            # 更新行数据
            row['sequence_item_ids'] = ','.join(sequence_item_ids)
            row['sequence_ratings'] = ','.join(sequence_ratings)
            row['sequence_timestamps'] = ','.join(sequence_timestamps)
            
            # 写入处理后的数据行
            writer.writerow(row)
    print(f"file saved at {outfile}")

# 示例使用
x = 5
# input_filename = '/home/xieminhui/yinj/workplace/gr-kvCache/generative-recommenders/test/data/test_eval_200.csv'  # 输入文件名
# output_filename = '/home/xieminhui/yinj/workplace/gr-kvCache/generative-recommenders/test/data/test_eval_200_loss_last_'+str(x)+'.csv'  # 输出文件名

# input_filename = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/generative-recommenders/test_data/ml_20m_sasrec_format_1000.csv'
# output_filename = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/generative-recommenders/test_data/ml_20m_sasrec_format_1000_loss_last_'+str(x)+'.csv'

# input_filename = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/generative-recommenders/test_data/ml_20m_sasrec_format_10000.csv'
# output_filename = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/generative-recommenders/test_data/ml_20m_sasrec_format_10000_loss_last_'+str(x)+'.csv'

# input_filename = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/generative-recommenders/test_data/ml_1m_sasrec_format_last_1000.csv'
# output_filename = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/generative-recommenders/test_data/ml_1m_sasrec_format_last_1000_loss_last_'+str(x)+'.csv'
# input_filename = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test.csv'
# output_filename = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_loss_last'+str(x)+'.csv'

# input_filename = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_head_1281.csv'
# output_filename = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_head_1281_loss_last_'+str(x)+'.csv'

# input_filename = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_head_1281_max_200.csv'
# output_filename = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_head_1281_max_200_loss_last_' + str(x) +'.csv'

input_filename = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_head_1281_max_200.csv'
output_filename = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_head_1281_max_200_loss_last_random_1_to_no_' + str(x) +'.csv'
# remove_last_x_sequences(input_filename, output_filename, 1, x - 1)

def compute_kv_diff(
    file_A,
    file_B,
    k_name,
    v_name,
    kv_name,
    x_lengths = torch.tensor([200, 196, 200,  84, 140, 200,  25, 200,  96, 200, 121,  26, 130, 111,
         19,  34, 200,  97, 200,  36,  69, 175,  93,  52,  40, 200, 200, 200,
        200,  76, 200,  49,  36, 200, 127,  91,  34,  37,  28,  33,  61, 200,
        200,  49, 200,  32,  27,  54,  98, 168, 200, 142, 200,  25,  38, 106,
         33, 200,  40, 150, 200,  96,  74, 200, 200,  73,  88,  32,  44,  80,
         52, 200,  75, 200, 200, 200,  32,  68, 200, 200,  73,  28, 156, 200,
        109, 200,  35,  49,  46, 200, 159, 101, 200,  48,  94, 200, 200, 102,
         20, 200,  19, 200, 102, 157,  23, 187, 200,  31,  48,  42, 200,  64,
         71,  76, 184, 200, 200, 200, 200,  70,  34,  71,  49,  87,  25,  22,
         95,  33, 200,  97,  20, 111, 151, 200,  24,  90,  29, 200,  21, 145,
         97, 200,  29,  20,  97,  19,  71,  96, 113,  68, 144, 200, 135, 147,
         27,  36,  40,  26,  26, 167, 175, 146, 172,  31,  20,  38, 139, 163,
        130,  24, 124,  32,  46, 122, 200, 135, 101, 125, 200,  49, 200, 200,
         21, 200, 133, 123,  88, 200, 200,  75,  36, 200,  21, 176,  35, 155,
         44, 155, 200, 200],dtype=int),
):
    # 加载文件 A 和 B
    file_A = torch.load(file_A)  # 形状为 8 个元素的 list，每个元素是一个四元组
    file_B = torch.load(file_B)  # 同样

    # 初始化保存每层差值的列表
    diff_k_all = []
    diff_v_all = []

    diff_kv_all = []

    # 遍历每一层进行计算
    for i in range(8):  # 对每个四元组（即每一层）进行处理
        v_A = file_A[i][0]  # (L, D)
        k_A = file_A[i][2]  # (B, N, D)
        v_B = file_B[i][0]  # (L, D)
        k_B = file_B[i][2]  # (B, N, D)

        # 计算 v 的差值：按元素计算 v_A 和 v_B 的差值的绝对值之和
        diff_v_layer = torch.sum(torch.abs(v_A - v_B), dim=1)  # 形状为 (L,)
        diff_v_all.append(diff_v_layer)  # 将结果添加到 diff_v_all 中

        # 计算 k 的差值：按元素计算 k_A 和 k_B 的差值的绝对值之和
        diff_k_layer = torch.sum(torch.abs(k_A - k_B), dim=2)  # 形状为 (B, N)
        result = []
        for i in range(len(x_lengths)):
            result.append(diff_k_layer[i, :x_lengths[i]])
        diff_k_layer = torch.cat(result)
        diff_k_all.append(diff_k_layer)  # 将结果添加到 diff_k_all 中

        diff_kv_layer = diff_k_layer + diff_v_layer
        diff_kv_all.append(diff_kv_layer)

    # 打印结果
    print(f"diff_v_all: {len(diff_v_all)} layers")
    print(f"diff_k_all: {len(diff_k_all)} layers")
    print(f"diff_kv_all: {len(diff_kv_all)} layers")
    print(f"diff_k_all[0] is {diff_k_all[0].shape}")
    print(f"diff_v_all[0] is {diff_v_all[0].shape}")
    print(f"diff_kv_all[0] is {diff_kv_all[0].shape}")

    kv_base_dir = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/diff_kv"
    diff_k_path = os.path.join(kv_base_dir, k_name)
    diff_v_path = os.path.join(kv_base_dir, v_name)
    diff_kv_path = os.path.join(kv_base_dir, kv_name)
    torch.save(diff_k_all, diff_k_path)
    torch.save(diff_v_all, diff_v_path)
    torch.save(diff_kv_all, diff_kv_path)
    print(f"diff_k saved at {diff_k_path}\ndiff_v saved at {diff_v_path}\ndiff_kv saved at {diff_kv_path}")

def draw_heatmap(
    diff_path,
    graph_name,
    type,
    random_size,
):
    diff_all = torch.load(diff_path)
    L = diff_all[0].shape[0]
    diff_matrix = torch.stack(diff_all, dim=1).cpu().numpy()

    if random_size == -1:
        random_size = L

    num_tokens = diff_matrix.shape[0]
    print(f"num_tokens is {num_tokens}")
    sampled_tokens = np.random.choice(num_tokens, size=random_size, replace=False)
    sampled_diff_matrix = diff_matrix[sampled_tokens]
    
    # 绘制热力图
    plt.figure(figsize=(12, 6))  # 设置图形大小
    # sns.heatmap(diff_matrix, cmap='coolwarm', annot=False, xticklabels=range(1, 9), yticklabels=range(1, L+1))
    plt.pcolormesh(sampled_diff_matrix, cmap='Blues', shading='auto')
    plt.colorbar()

    # 添加标题和标签
    plt.title(f"Heatmap of diff_{type} across Layers and Tokens")
    plt.xlabel("Layers")
    plt.ylabel("Tokens")

    base_dir = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/graphs"
    file_name = graph_name + type + "_" + str(random_size) + ".png"
    filepath = os.path.join(base_dir, file_name)
    plt.savefig(filepath)
    print(f"diff_heatmap saved at {filepath}")

diff_k_path = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/diff_kv/base_step1_diff_k.pt"
diff_v_path = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/diff_kv/base_step1_diff_v.pt"
diff_kv_path = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/diff_kv/base_step1_diff_kv.pt"
graph_name = "base_step1_diff_"
# draw_heatmap(diff_k_path, graph_name, 'k', 100)
# draw_heatmap(diff_v_path, graph_name, 'v', 100)
# draw_heatmap(diff_k_path, graph_name, 'k', -1)
# draw_heatmap(diff_v_path, graph_name, 'v', -1)
# draw_heatmap(diff_kv_path, graph_name, 'kv', 30)
# draw_heatmap(diff_kv_path, graph_name, 'kv', 50)
# draw_heatmap(diff_kv_path, graph_name, 'kv', 100)
# draw_heatmap(diff_kv_path, graph_name, 'kv', 150)
# draw_heatmap(diff_kv_path, graph_name, 'kv', 200)
# draw_heatmap(diff_kv_path, graph_name, 'kv', -1)
# draw_heatmap(diff_kv_path, graph_name, 'kv', 1000)
# draw_heatmap(diff_kv_path, graph_name, 'kv', 2000)
# draw_heatmap(diff_kv_path, graph_name, 'kv', 5000)

def draw_heatmap_pearsonr_workflow(
    model_a_path: str,
    model_b_path: str,
    output_base_name: str,
    x_lengths: torch.Tensor,
    random_size: int = 100
):
    """
    完整工作流：计算模型差异 → 生成热力图 → 绘制相关性图
    
    参数:
        model_a_path: 模型A的kvc文件路径
        model_b_path: 模型B的kvc文件路径
        output_base_name: 输出文件基础名称 (如 "base_step1")
        x_lengths: 序列长度张量
        random_size: 热力图采样数量 (默认100)
    """
    # 1. 定义输出路径
    kv_base_dir = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/diff_kv"
    graph_base_dir = "/data/xieminhui/yinj/workplace/gr-kvCache/newGR/graphs"
    
    # 2. 计算kv差异
    def _compute_diff():
        k_name = f"{output_base_name}_diff_k.pt"
        v_name = f"{output_base_name}_diff_v.pt"
        kv_name = f"{output_base_name}_diff_kv.pt"
        
        compute_kv_diff(
            file_A=model_a_path,
            file_B=model_b_path,
            k_name=k_name,
            v_name=v_name,
            kv_name=kv_name,
            x_lengths=x_lengths  # 需要修改原compute_kv_diff函数接受此参数
        )
        return (
            os.path.join(kv_base_dir, k_name),
            os.path.join(kv_base_dir, v_name),
            os.path.join(kv_base_dir, kv_name)
        )

    diff_k_path, diff_v_path, diff_kv_path = _compute_diff()

    # 3. 绘制热力图
    def _draw_heatmaps():
        for diff_type, path in zip(['k', 'v', 'kv'], [diff_k_path, diff_v_path, diff_kv_path]):
            draw_heatmap(
                diff_path=path,
                graph_name=f"{output_base_name}_",
                type=diff_type,
                random_size=random_size
            )
            draw_heatmap(
                diff_path=path,
                graph_name=f"{output_base_name}_",
                type=diff_type,
                random_size=-1
            )

    _draw_heatmaps()

    # 4. 绘制皮尔森相关性图
    def _plot_correlation():
        plot_kv_diff_correlation(
            kv_diff_file=diff_kv_path,
            graph_name=output_base_name
        )

    _plot_correlation()

base_path = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/cached_uvqk/saved_cache_with_ckpt_base_train_5000_ep20'
step_path = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/cached_uvqk/saved_cache_with_ckpt_delta_128_step_1'
step_2_path = '/data/xieminhui/yinj/workplace/gr-kvCache/newGR/data/cached_uvqk/saved_cache_with_ckpt_delta_128_step_2'
k_name = 'base_step1_diff_k.pt'
v_name = 'base_step1_diff_v.pt'
kv_name = 'base_step1_diff_kv.pt'
# compute_kv_diff(base_path, step_path, k_name, v_name, kv_name)

base_cache_list_path='/home/yinj@/datas/grkvc/base_cache_and_cached_lengths/base_cache_list.pt'
cached_lengths_list_path='/home/yinj@/datas/grkvc/base_cache_and_cached_lengths/cached_lengths_list.pt'
# base_cache_list = torch.load(base_cache_list_path)

# cached_lengths_list = torch.load(cached_lengths_list_path)
# x_lengths = cached_lengths_list[0]

# ml-20m test data(/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_loss_last5.csv) iter 1
x_lengths = torch.tensor([ 43,  59,  36, 200,  13, 181, 104, 123,  36,  33,  63,  66, 200,  96,
         15,  26, 191, 200, 200,  43,  53, 200, 110, 127, 183,  15,  51,  16,
        200, 126,  53,  26,  15,  13,  43, 127,  24,  37,  14,  15,  67,  76,
         17,  83, 193,  67,  31,  16,  16,  32, 200, 200, 200, 111,  15,  38,
        200,  56, 200, 200, 154,  32, 200, 200, 190,  28,  32,  22, 162,  13,
         71,  14,  84,  18, 200, 105,  79,  25, 200,  17,  78, 200,  39, 116,
        200, 200, 200,  79,  54, 129,  78,  50, 140,  83, 110,  45, 200,  36,
         33,  23,  78,  13, 164,  52,  30,  46,  15,  84,  54,  83, 200, 118,
         16,  96, 122,  13,  49, 108,  30, 200,  40,  68,  26, 158,  54, 200,
        200, 154])
base_cache_path='/home/yinj@/datas/grkvc/base_cache_and_cached_lengths/base_model_iter_1_cache.pt'

# torch.save(base_cache_list[0], base_cache_path)
# print(f"base_cache_list[0] saved at {base_cache_path}")

# torch.save(base_cache_list[1], base_cache_path)
# print(f"base_cache_list[0] saved at {base_cache_path}")




x_lengths = torch.tensor([200, 196, 200,  84, 140, 200,  25, 200,  96, 200, 121,  26, 130, 111,
         19,  34, 200,  97, 200,  36,  69, 175,  93,  52,  40, 200, 200, 200,
        200,  76, 200,  49,  36, 200, 127,  91,  34,  37,  28,  33,  61, 200,
        200,  49, 200,  32,  27,  54,  98, 168, 200, 142, 200,  25,  38, 106,
         33, 200,  40, 150, 200,  96,  74, 200, 200,  73,  88,  32,  44,  80,
         52, 200,  75, 200, 200, 200,  32,  68, 200, 200,  73,  28, 156, 200,
        109, 200,  35,  49,  46, 200, 159, 101, 200,  48,  94, 200, 200, 102,
         20, 200,  19, 200, 102, 157,  23, 187, 200,  31,  48,  42, 200,  64,
         71,  76, 184, 200, 200, 200, 200,  70,  34,  71,  49,  87,  25,  22,
         95,  33, 200,  97,  20, 111, 151, 200,  24,  90,  29, 200,  21, 145,
         97, 200,  29,  20,  97,  19,  71,  96, 113,  68, 144, 200, 135, 147,
         27,  36,  40,  26,  26, 167, 175, 146, 172,  31,  20,  38, 139, 163,
        130,  24, 124,  32,  46, 122, 200, 135, 101, 125, 200,  49, 200, 200,
         21, 200, 133, 123,  88, 200, 200,  75,  36, 200,  21, 176,  35, 155,
         44, 155, 200, 200],dtype=int)

# draw_heatmap_pearsonr_workflow(model_a_path=base_path,model_b_path=step_2_path,output_base_name="base_step2",x_lengths=x_lengths,random_size=150,)