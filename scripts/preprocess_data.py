import bisect
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch
import re
import csv
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.stats import pearsonr

def convert_to_timestamp(dates):
    """
    将日期列表转换为 Unix 时间戳
    :param dates: 日期列表，格式为['YYYY-MM-DD', ...]
    :return: 时间戳列表
    """
    timestamps = []
    for date in dates:
        timestamp = int(datetime.strptime(date + " 00:00:00", "%Y-%m-%d %H:%M:%S").timestamp())
        timestamps.append(timestamp)
    return timestamps

def process_data_by_timestamps(csv_file, dates):
    """
    读取csv文件，按照时间戳截断数据并保存每个时间戳的数据为单独的CSV文件
    :param csv_file: 输入的CSV文件路径
    :param timestamps: 时间戳列表
    :param output_dir: 输出文件夹路径
    """
    timestamps = convert_to_timestamp(dates=dates)
    # 读取csv文件
    df = pd.read_csv(csv_file)
    
    # 处理每个时间戳
    for timestamp in timestamps:
        df_processed = []
        
        # 对每一行进行处理
        for _, row in df.iterrows():
            # 获取每个字段
            user_id = row['user_id']
            sequence_item_ids = row['sequence_item_ids']
            sequence_ratings = row['sequence_ratings']
            sequence_timestamps = row['sequence_timestamps']
            
            # 将这些列转换为列表
            item_ids = list(map(int, sequence_item_ids.split(',')))
            ratings = list(map(float, sequence_ratings.split(',')))
            timestamps = list(map(int, sequence_timestamps.split(',')))
            
            # 截断数据，保留小于等于当前时间戳的数据
            valid_indices = [i for i, ts in enumerate(timestamps) if ts <= timestamp]
            truncated_item_ids = [item_ids[i] for i in valid_indices]
            truncated_ratings = [ratings[i] for i in valid_indices]
            truncated_timestamps = [timestamps[i] for i in valid_indices]
            
            # 保证处理后的数据长度与原数据一致，填充不足的部分
            padding_length = max(len(truncated_item_ids), len(item_ids))  # 保持长度一致
            truncated_item_ids += [0] * (padding_length - len(truncated_item_ids))
            truncated_ratings += [0] * (padding_length - len(truncated_ratings))
            truncated_timestamps += [0] * (padding_length - len(truncated_timestamps))
            
            # 更新处理后的数据
            row['sequence_item_ids'] = ','.join(map(str, truncated_item_ids))
            row['sequence_ratings'] = ','.join(map(str, truncated_ratings))
            row['sequence_timestamps'] = ','.join(map(str, truncated_timestamps))
            
            # 将处理后的行添加到新的数据集中
            df_processed.append(row)
        
        # 创建输出文件名
        file_dir, file_name = os.path.split(csv_file)
        file_base, file_ext = os.path.splitext(file_name)
        date = datetime.utcfromtimestamp(timestamp).strftime('%y-%m-%d')
        output_file = os.path.join(file_dir, f"{file_base}_{date}.csv")
        
        # 保存截断后的数据到文件
        df_processed_df = pd.DataFrame(df_processed)
        df_processed_df.to_csv(output_file, index=False)
        print(f"Saved data for timestamp {timestamp} to {output_file}")

def extract_metrics_from_log(log_file_path, output_csv_path):# with timestamp format
    """
    该函数从日志文件中提取时间戳和指标数据，并将其保存到 CSV 文件中。

    参数:
    log_file_path (str): 日志文件的路径
    output_csv_path (str): 输出 CSV 文件的路径
    """
    # 读取日志文件
    with open(log_file_path, 'r') as file:
        log_data = file.readlines()
    
    # 正则表达式匹配时间戳和指标数据
    timestamp_pattern = r"model_ep20_(\d+)\.ckpt"
    metrics_pattern = r"metrics are: NDCG@10 (\d+\.\d+), NDCG@50 (\d+\.\d+), HR@10 (\d+\.\d+), HR@50 (\d+\.\d+), MRR (\d+\.\d+)"
    
    # 用于存储提取的数据
    extracted_data = []
    
    # 遍历日志行
    for line in log_data:
        # 检查行中是否有时间戳
        timestamp_match = re.search(timestamp_pattern, line)
        if timestamp_match:
            timestamp = int(timestamp_match.group(1))
            # 将时间戳转换为日期
            date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
        
        # 检查行中是否有指标数据
        metrics_match = re.search(metrics_pattern, line)
        if metrics_match:
            NDCG10 = float(metrics_match.group(1))
            NDCG50 = float(metrics_match.group(2))
            HR10 = float(metrics_match.group(3))
            HR50 = float(metrics_match.group(4))
            MRR = float(metrics_match.group(5))
            
            # 将提取的数据添加到列表中
            extracted_data.append([date, NDCG10, NDCG50, HR10, HR50, MRR])
    
    # 创建 DataFrame 并保存为 CSV 文件
    df = pd.DataFrame(extracted_data, columns=["Date", "NDCG@10", "NDCG@50", "HR@10", "HR@50", "MRR"])
    df.to_csv(output_csv_path, index=False)
    return f"Data saved successfully to {output_csv_path}"

def truncate_sequence_data(inputfile, max_seq_len):
    file_path = inputfile
    file_dir, file_name = os.path.split(inputfile)
    file_base, file_ext = os.path.splitext(file_name)
    output_file = os.path.join(file_dir, f"{file_base}_max_{max_seq_len}.csv")
    df = pd.read_csv(file_path)

    def truncate_sequences(row, max_seq_len):
        # 将字符串按逗号分割成列表
        item_ids = row['sequence_item_ids'].split(',')
        ratings = row['sequence_ratings'].split(',')
        timestamps = row['sequence_timestamps'].split(',')

        # 检查列表长度是否大于等于max_seq_len
        if len(item_ids) >= max_seq_len:
            # 截取前max_seq_len个元素
            item_ids = item_ids[:max_seq_len]
            ratings = ratings[:max_seq_len]
            timestamps = timestamps[:max_seq_len]

            # 将处理后的列表重新拼接成字符串
            row['sequence_item_ids'] = ','.join(item_ids)
            row['sequence_ratings'] = ','.join(ratings)
            row['sequence_timestamps'] = ','.join(timestamps)

        return row
    df = df.apply(truncate_sequences, args=(max_seq_len, ), axis=1)

    df.to_csv(output_file, index=False)
    print(f"data truncated by {max_seq_len} saved at {output_file}")

def convert_log_to_csv(input_file):
    output_file = input_file.replace('.log', '.csv')

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    def extract_metrics(log_content):
        # 匹配检查点块的正则表达式
        checkpoint_pattern = r'\[loading checkpoint\] from: (.*?)\n(.*?)(?=\n\*{80}|\Z)'
        # checkpoint_pattern = r'\[loading checkpoint\] from: (.*?)\n(.*?)(?=\n[\*]{3,}|\n={3,}|\Z)'
        checkpoint_blocks = re.findall(checkpoint_pattern, log_content, re.DOTALL)

        data = []
        for ckpt_path, block_content in checkpoint_blocks:
            ckpt_file = os.path.basename(ckpt_path.strip())
            model_name = os.path.splitext(ckpt_file)[0]

            # 匹配普通缓存类型（no/fully）的 avg_time 和 metrics
            eval_pattern = r'''
                =+\s+begin\s+evaling\s+use\s+(\w+)\s+cache.*?=\s*\n  # 匹配缓存类型
                .*?eval\s+use\s+\w+\s+cache\s+need\s+avg\s+:\s+([\d.]+)\s+ms\n  # 提取 avg_time
                .*?metrics\s+are:\s+(.*?)\n  # 提取指标
            '''
            eval_matches = re.findall(eval_pattern, block_content, re.DOTALL | re.VERBOSE)

            # 匹配 selective 缓存类型
            selective_pattern = r'''
                !!!recompute_ratio\s+is\s+(\d+)!!!  # 提取重计算比例
                .*?eval\s+use\s+selective\s+cache\s+need\s+avg\s+:\s+([\d.]+)\s+ms\n  # 提取 avg_time
                .*?metrics\s+are:\s+(.*?)\n  # 提取指标
            '''
            selective_matches = re.findall(selective_pattern, block_content, re.DOTALL | re.VERBOSE)

            # 处理 no/fully 类型
            for match in eval_matches:
                cache_type, avg_time_str, metrics_str = match
                if cache_type == 'selective':
                    continue
                
                metrics = extract_metrics_pairs(metrics_str)
                data.append({
                    'model_name': model_name,
                    'cache_type': cache_type,
                    'recompute_ratio': 0,
                    'avg_time(ms)': float(avg_time_str),
                    **metrics
                })

            # 处理 selective 类型
            for ratio, avg_time_str, metrics_str in selective_matches:
                metrics = extract_metrics_pairs(metrics_str)
                data.append({
                    'model_name': model_name,
                    'cache_type': 'selective',
                    'recompute_ratio': int(ratio),
                    'avg_time(ms)': float(avg_time_str),
                    **metrics
                })

        return data

    def extract_metrics_pairs(metrics_str):
        metrics = {}
        metrics_pairs = re.findall(r'([A-Z]+@\d+|\bMRR\b)\s+([0-9.]+)', metrics_str)
        for key, value in metrics_pairs:
            metrics[key] = float(value)
        return metrics

    metrics_data = extract_metrics(content)
    
    if not metrics_data:
        print("No valid data found.")
        return
    
    # 动态获取字段（包含 avg_time）
    fieldnames = ['model_name', 'cache_type', 'recompute_ratio', 'avg_time(ms)']
    metric_fields = sorted({k for row in metrics_data for k in row.keys() if k not in fieldnames})
    fieldnames += metric_fields
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_data)
    
    print(f"Metrics saved at {output_file}")
    return output_file

def plot_metrics_from_csv(csv_file, output_dir='/home/yinj@/datas/grkvc/graphs', output_name='metrics_comparison_line.png'):
    """
    绘制推荐系统指标对比图并保存
    
    Parameters:
    csv_file (str): 输入CSV文件路径，需包含model_name和各指标列
    output_dir (str): 输出目录，默认当前目录
    output_name (str): 输出图片文件名，默认metrics_comparison.png
    
    Returns:
    str: 保存成功的完整路径信息
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"csv file is not exist{csv_file}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取数据
        df = pd.read_csv(csv_file)
        
        # 校验必要字段
        required_columns = ['model_name', 'NDCG@10', 'NDCG@50', 'HR@10', 'HR@50', 'MRR']
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"CSV file: {csv_file} miss column {missing}")

        # 设置画布
        plt.figure(figsize=(25, 5))
        metrics = ['NDCG@10', 'NDCG@50', 'HR@10', 'HR@50', 'MRR']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        x_ticks = range(len(df))  # 所有数据点的索引

        # 绘制子图
        for i, metric in enumerate(metrics, 1):
            ax = plt.subplot(1, 5, i)
            
            # 绘制折线（不再依赖df.plot自动处理x轴）
            ax.plot(
                x_ticks,
                df[metric],
                marker='o',
                color=colors[i-1],
                linewidth=2,
                markersize=8
            )
            
            # 显式设置x轴
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(
                df['model_name'],
                rotation=45,
                ha='right',
                fontsize=10
            )
            
            # 配置其他元素
            ax.set_title(metric, fontsize=14, pad=15)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 动态调整y轴范围
            buffer = 0.02
            y_min = df[metric].min() - buffer
            y_max = df[metric].max() + buffer
            ax.set_ylim(max(y_min, 0), min(y_max, 1.0))  # 确保不超出[0,1]范围

        # 调整布局并保存
        plt.tight_layout(pad=3.0)
        output_path = os.path.join(output_dir, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"metrics line plot saved at: {os.path.abspath(output_path)}")

    except Exception as e:
        plt.close()  # 确保关闭画布
        print(f"plot failure! {str(e)}")

def plot_selective_metrics_from_csv(csv_file, output_dir='/home/yinj@/datas/grkvc/graphs', output_name='metrics_comparison_line.png'):
    """
    绘制推荐系统指标对比图并保存
    
    Parameters:
    csv_file (str): 输入CSV文件路径，需包含model_name和各指标列
    output_dir (str): 输出目录，默认当前目录
    output_name (str): 输出图片文件名，默认metrics_comparison.png
    
    Returns:
    str: 保存成功的完整路径信息
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"csv file is not exist{csv_file}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取数据
        df = pd.read_csv(csv_file)
        
        # 校验必要字段
        required_columns = ['model_name', 'NDCG@10', 'HR@10', 'MRR']
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"CSV file: {csv_file} miss column {missing}")

        # 设置画布
        plt.figure(figsize=(15, 5))
        metrics = ['NDCG@10', 'HR@10', 'MRR']
        colors = ['#1f77b4', '#2ca02c', '#9467bd']
        x_ticks = range(len(df))  # 所有数据点的索引

        # 绘制子图
        for i, metric in enumerate(metrics, 1):
            ax = plt.subplot(1, 3, i)
            
            # 绘制折线（不再依赖df.plot自动处理x轴）
            ax.plot(
                x_ticks,
                df[metric],
                marker='o',
                color=colors[i-1],
                linewidth=2,
                markersize=8
            )
            
            # 显式设置x轴
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(
                df['model_name'],
                rotation=45,
                ha='right',
                fontsize=10
            )
            
            # 配置其他元素
            ax.set_title(metric, fontsize=14, pad=15)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 动态调整y轴范围
            buffer = 0.02
            y_min = df[metric].min() - buffer
            y_max = df[metric].max() + buffer
            ax.set_ylim(max(y_min, 0), min(y_max, 1.0))  # 确保不超出[0,1]范围

        # 调整布局并保存
        plt.tight_layout(pad=3.0)
        output_path = os.path.join(output_dir, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"metrics line plot saved at: {os.path.abspath(output_path)}")

    except Exception as e:
        plt.close()  # 确保关闭画布
        print(f"plot failure! {str(e)}")

def filter_rows_by_seq_len(inputfile, max_seq_len, item_column='sequence_item_ids'):
    file_dir, file_name = os.path.split(inputfile)
    file_base, file_ext = os.path.splitext(file_name)
    output_file = os.path.join(file_dir, f"{file_base}_full_{max_seq_len}.csv")
    df = pd.read_csv(inputfile)
    
    # 定义一个内部函数来计算逗号分隔的元素个数
    def count_items(item_list):
        return len(item_list.split(','))

    # 应用这个函数到指定列，并筛选出元素个数等于目标值的行
    df_filtered = df[df[item_column].apply(count_items) == max_seq_len]

    # 保存筛选后的DataFrame到CSV
    df_filtered.to_csv(output_file, index=False)

    print(f"Filtered data saved to {output_file}") 

def plot_1_model_selective_metrics_from_full_test_csv(csv_file, model_name, output_dir='/home/yinj@/datas/grkvc/graphs', image_prefix='metrics'):
    """
    绘制指定模型的性能指标和推理时间图表，并保存为图片。

    参数:
    - df: DataFrame，包含模型的性能数据。
    - model_name: str，要绘制的模型名称。
    - image_prefix: str，保存图片的前缀名称。
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"csv file is not exist{csv_file}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取数据
        df = pd.read_csv(csv_file)
        # 筛选特定模型版本
        df_model = df[df["model_name"] == model_name]

        # 创建画布和子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # 绘制性能指标（HR@10）
        for cache_type in df_model["cache_type"].unique():
            df_cache = df_model[df_model["cache_type"] == cache_type]
            ax1.plot(
                df_cache["recompute_ratio"],
                df_cache["HR@10"],
                label=f"{cache_type} cache",
                marker="o" if cache_type == "selective" else None,
            )

        ax1.set_title(f"HR@10 vs Recompute Ratio ({model_name})")
        ax1.set_xlabel("Recompute Ratio")
        ax1.set_ylabel("HR@10")
        ax1.legend()
        ax1.grid(True)

        # 绘制推理时间（avg_time）
        for cache_type in df_model["cache_type"].unique():
            df_cache = df_model[df_model["cache_type"] == cache_type]
            ax2.plot(
                df_cache["recompute_ratio"],
                df_cache["avg_time(ms)"],
                label=f"{cache_type} cache",
                marker="o" if cache_type == "selective" else None,
            )

        ax2.set_title(f"Avg Time vs Recompute Ratio ({model_name})")
        ax2.set_xlabel("Recompute Ratio")
        ax2.set_ylabel("Avg Time (ms)")
        ax2.legend()
        ax2.grid(True)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        image_filename = f"{image_prefix}_{model_name}.png"
        plt.savefig(image_filename)

        print(f"Image saved as {image_filename}")
    except Exception as e:
        plt.close()  # 确保关闭画布
        print(f"plot failure! {str(e)}")

def plot_metrics(csv_file, selected_metrics, skip_model, output_dir='/home/yinj@/datas/grkvc/graphs', image_prefix='hahaha', y_lim=False):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 获取所有的模型名称，排除skip_model中的模型
    model_names = df['model_name'].unique()
    model_names = [model for model in model_names if model not in skip_model]
    
    # 1. 绘制不同recompute_ratio指标对比图，排布成一张图，4行3列
    fig, axes = plt.subplots(len(model_names), len(selected_metrics), figsize=(15, 5 * len(model_names)))
    axes = axes.flatten()  # 将 2D 数组展平为 1D 方便索引

    # 为每个指标计算全局的纵坐标范围
    global_y_limits = {}
    if y_lim:
        for metric in selected_metrics:
            global_y_min, global_y_max = None, None
            
            # 查找每个指标的最小值和最大值
            for model_name in model_names:
                df_model = df[df["model_name"] == model_name]
                df_cache_selective = df_model[df_model["cache_type"] == "selective"]
                
                metric_values = df_cache_selective[metric].values
                
                # 获取当前指标的最小值和最大值
                min_value = metric_values.min()
                max_value = metric_values.max()

                # 更新全局最小值和最大值
                if global_y_min is None or min_value < global_y_min:
                    global_y_min = min_value
                if global_y_max is None or max_value > global_y_max:
                    global_y_max = max_value

            global_y_limits[metric] = (global_y_min, global_y_max)

    for idx, model_name in enumerate(model_names):
        df_model = df[df["model_name"] == model_name]
        
        for i, metric in enumerate(selected_metrics):
            ax = axes[idx * len(selected_metrics) + i]  # 获取对应位置的子图
            legend_added = False  # 用于控制图例添加
            # 只绘制selective的线
            df_cache_selective = df_model[df_model["cache_type"] == "selective"]
            ax.plot(
                df_cache_selective["recompute_ratio"],
                df_cache_selective[metric],
                label=f"selective cache",
                marker="o",
            )

            # 添加两条水平虚线，代表no和fully对应的指标
            no_value = df_model[df_model["cache_type"] == "no"][metric].iloc[0]
            fully_value = df_model[df_model["cache_type"] == "fully"][metric].iloc[0]
            middle_value = fully_value + 0.9 * (no_value - fully_value)

            ax.axhline(no_value, color='red', linestyle='--', label="no cache")
            ax.axhline(fully_value, color='blue', linestyle='--', label="fully cache")
            ax.axhline(middle_value, color='green', linestyle='--', label="90% between no and fully")

            ax.set_title(f"{metric} vs Recompute Ratio ({model_name})")
            ax.set_xlabel("Recompute Ratio")
            ax.set_ylabel(metric)
            ax.grid(True)

            # 设置纵坐标范围
            if y_lim:
                y_min, y_max = global_y_limits[metric]
                ax.set_ylim(y_min, y_max)  # 设置该指标的纵坐标范围

            ax.legend()

    # 保存图片
    if y_lim == True:
        image_filename = os.path.join(output_dir, f"{image_prefix}_wYLim_recompute_ratio_metrics.png")
    else:
        image_filename = os.path.join(output_dir, f"{image_prefix}_recompute_ratio_metrics.png")
    plt.tight_layout()
    plt.savefig(image_filename)
    plt.close()
    print(f"Saved {image_filename}")

    # 2. 绘制不同recompute_ratio时间对比图，排布成1行4列
    fig, axes = plt.subplots(1, len(model_names), figsize=(20, 5))  # 1 行 4 列
    for i, model_name in enumerate(model_names):
        df_model = df[df["model_name"] == model_name]
        ax = axes[i]  # 选择对应的子图位置
        legend_added = False  # 用于控制图例添加
        # 只绘制selective的线
        df_cache_selective = df_model[df_model["cache_type"] == "selective"]
        ax.plot(
            df_cache_selective["recompute_ratio"],
            df_cache_selective["avg_time(ms)"],
            label=f"selective cache",
            marker="o",
        )

        # 添加两条水平虚线，代表no和fully对应的avg_time
        no_avg_time = df_model[df_model["cache_type"] == "no"]["avg_time(ms)"].iloc[0]
        fully_avg_time = df_model[df_model["cache_type"] == "fully"]["avg_time(ms)"].iloc[0]
        ax.axhline(no_avg_time, color='red', linestyle='--', label="no cache")
        ax.axhline(fully_avg_time, color='blue', linestyle='--', label="fully cache")

        ax.set_title(f"Avg Time vs Recompute Ratio ({model_name})")
        ax.set_xlabel("Recompute Ratio")
        ax.set_ylabel("Avg Time (ms)")
        ax.grid(True)

        ax.legend()

    # 保存图片
    image_filename = os.path.join(output_dir, f"{image_prefix}_recompute_ratio_time.png")
    plt.tight_layout()
    plt.savefig(image_filename)
    plt.close()
    print(f"Saved {image_filename}")

    # 3. 绘制完全复用精度损失图，排布成1行3列
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 行 3 列
    for i, metric in enumerate(selected_metrics):
        ax = axes[i]
        
        width = 0.35  # 设置柱状图的宽度，确保柱子之间正常相邻
        offset = width/2  # 设置偏移量为0，确保柱子紧挨
        
        legend_added = False  # 用于控制图例添加
        
        for j, model_name in enumerate(model_names):
            df_model = df[df["model_name"] == model_name]
            # 提取 no 和 fully 的指标数据
            no_metric_value = df_model[df_model["cache_type"] == "no"][metric].iloc[0]
            fully_metric_value = df_model[df_model["cache_type"] == "fully"][metric].iloc[0]
            
            # 获取日期部分
            model_date = model_name.split('_')[-2]  # 假设日期在模型名的最后部分，格式如 model_02-02
            x_position = j  # 偏移量，确保柱状图不重叠
            
            # 绘制柱状图
            ax.bar(x_position - offset, no_metric_value, width=width, label="no cache", align="center", color='#4C72B0')
            ax.bar(x_position + offset, fully_metric_value, width=width, label="fully cache", align="center", color='#DD8452')

            # 只在第一次添加图例时才显示
            if not legend_added:
                ax.legend(["no cache", "fully cache"])
                legend_added = True

        ax.set_title(f"{metric} Loss vs Model Name")
        ax.set_xlabel("Model Name")
        ax.set_ylabel(f"{metric}")
        ax.set_xticks(range(len(model_names)))  # 横坐标位置
        ax.set_xticklabels([model_name.split('_')[-2] for model_name in model_names])  # 横坐标标签为日期
        ax.grid(True, linestyle='--', alpha=0.35)

    # 保存图片
    image_filename = os.path.join(output_dir, f"{image_prefix}_loss_comparison.png")
    plt.tight_layout()
    plt.savefig(image_filename)
    plt.close()
    print(f"Saved {image_filename}")

def compute_kv_diff(file1, file2, length_file, output_dir, output_prefix):
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载两个KV cache文件
    kv_cache1 = torch.load(file1)
    kv_cache2 = torch.load(file2)

    # 加载有效缓存长度
    cache_lengths = torch.load(length_file)

    # 初始化存储差异的列表
    k_diff_list = []
    v_diff_list = []
    kv_diff_list = []

    # 遍历每一层的KV cache
    for layer_idx in range(len(kv_cache1)):
        # 获取当前层的K和V
        v1, k1 = kv_cache1[layer_idx][0], kv_cache1[layer_idx][2]
        v2, k2 = kv_cache2[layer_idx][0], kv_cache2[layer_idx][2]

        # 计算K和V的差异（欧氏距离）
        k_diff = torch.norm(k1 - k2, dim=2)  # 形状为 (B, N)
        v_diff = torch.norm(v1 - v2, dim=2)  # 形状为 (B, N)
        kv_diff = k_diff + v_diff  # 形状为 (B, N)

        # 根据有效缓存长度拼接所有用户的差异值
        layer_k_diff = []
        layer_v_diff = []
        layer_kv_diff = []
        for b in range(k1.size(0)):
            layer_k_diff.append(k_diff[b, :cache_lengths[b]])
            layer_v_diff.append(v_diff[b, :cache_lengths[b]])
            layer_kv_diff.append(kv_diff[b, :cache_lengths[b]])

        # 将当前层的差异值拼接为一个形状为 (L,) 的张量
        layer_k_diff = torch.cat(layer_k_diff)
        layer_v_diff = torch.cat(layer_v_diff)
        layer_kv_diff = torch.cat(layer_kv_diff)

        # 将当前层的差异值添加到列表中
        k_diff_list.append(layer_k_diff)
        v_diff_list.append(layer_v_diff)
        kv_diff_list.append(layer_kv_diff)

    # 将差异值保存到文件中
    k_save_path = os.path.join(output_dir, f"{output_prefix}_k_diff.pt")
    v_save_path = os.path.join(output_dir, f"{output_prefix}_v_diff.pt")
    kv_save_path = os.path.join(output_dir, f"{output_prefix}_kv_diff.pt")
    torch.save(k_diff_list, k_save_path)
    torch.save(v_diff_list, v_save_path)
    torch.save(kv_diff_list, kv_save_path)

    print(f"Saved k_diff to {k_save_path}")
    print(f"Saved v_diff to {v_save_path}")
    print(f"Saved kv_diff to {kv_save_path}")

def plot_diff_heatmap(diff_file_path, output_dir, output_prefix, token_start, token_end, title=None):
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载差异文件
    diff_data = torch.load(diff_file_path)  # 形状为 (16, 12672)

    # 将每层的差异值堆叠成一个形状为 (16, 12672) 的张量
    diff_tensor = torch.stack(diff_data)  # 形状为 (16, 12672)

    # 提取指定范围内的 token 数据
    heatmap_data = diff_tensor[:, token_start:token_end]  # 形状为 (16, token_end - token_start)

    # 将数据转换为 NumPy 数组
    heatmap_data_np = heatmap_data.numpy()

    # 创建热力图
    plt.figure(figsize=(10, 6))
    plt.imshow(
        heatmap_data_np,
        cmap='Blues',  # 使用蓝色调色板
        aspect='auto',  # 自动调整纵横比
        vmin=np.min(heatmap_data_np),  # 颜色映射的最小值
        vmax=np.max(heatmap_data_np),  # 颜色映射的最大值
        alpha=0.8  # 设置透明度
    )
    plt.colorbar(label='Diff Value')  # 添加颜色条
    plt.title(title or f'Diff Heatmap (Tokens {token_start} to {token_end})')
    plt.xlabel('Token Index')
    plt.ylabel('Layer ID')

    # 设置横轴和纵轴刻度
    num_tokens = token_end - token_start
    xticks = range(0, num_tokens, max(1, num_tokens // 10))  # 横轴刻度
    xtick_labels = range(token_start, token_end, max(1, num_tokens // 10))  # 横轴刻度标签
    plt.xticks(xticks, labels=xtick_labels)  # 设置横轴刻度和标签

    yticks = range(len(diff_data))  # 纵轴刻度
    ytick_labels = range(len(diff_data))  # 纵轴刻度标签
    plt.yticks(yticks, labels=ytick_labels)  # 设置纵轴刻度和标签

    # 保存热力图
    output_path = os.path.join(output_dir, f"{output_prefix}_heatmap_token{token_start}_to_{token_end}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved cheatmap to {output_path}")

def plot_layer_correlation(diff_file_path, output_dir, output_prefix, threshold=None, title=None):
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载差异文件
    diff_data = torch.load(diff_file_path)  # 形状为 (16, L)

    # 将每层的差异值堆叠成一个形状为 (16, L) 的张量
    diff_tensor = torch.stack(diff_data)  # 形状为 (16, L)

    # 计算相邻层的皮尔森相关性系数
    correlations = []
    layer_pairs = []
    for i in range(len(diff_tensor) - 1):
        layer1 = diff_tensor[i].numpy()  # 当前层
        layer2 = diff_tensor[i + 1].numpy()  # 下一层
        corr, _ = pearsonr(layer1, layer2)  # 计算皮尔森相关性系数
        correlations.append(corr)
        layer_pairs.append(f"{i+1} vs {i+2}")  # 记录相邻层的标签

    # 创建相关性系数图
    plt.figure(figsize=(10, 6))
    plt.bar(
        layer_pairs, 
        correlations, 
        color='#1f77b4',  # 浅蓝色
        alpha=0.8  # 设置透明度
    )
    plt.title(title or 'Layer-wise Pearson Correlation')
    plt.xlabel('Adjacent Layers')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.ylim(0, 1)  # 设置纵轴范围为 [0, 1]

    # 绘制虚线（如果传入了阈值）
    if threshold is not None:
        plt.axhline(
            threshold, 
            color='red',  # 灰色
            linestyle='--',  # 虚线
            linewidth=1.5,  # 线宽
            label=f'Threshold: {threshold}'  # 图例标签
        )
        plt.legend()  # 显示图例

    # 保存图
    output_path = os.path.join(output_dir, f"{output_prefix}_layer_correlation.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"layer correlateion saved {output_path}")

def plot_kv_diff_cdf(diff_file_path, output_dir, output_prefix, threshold=None, title=None):
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载差异文件
    kv_diff_list = torch.load(diff_file_path)  # 形状为 (16, L)，每个元素为一个tensor
    
    # 创建颜色映射（蓝色渐变）
    cmap = plt.cm.Blues  # 蓝色渐变色
    norm = plt.Normalize(vmin=0, vmax=len(kv_diff_list) - 1)  # 正常化颜色范围

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))  # 使用fig, ax创建图形
    
    # 绘制每层的CDF
    for i, kv_diff in enumerate(kv_diff_list):
        # 将每层的差异值合并成一个1D张量
        all_kv_diffs = kv_diff.numpy()  # 转换为numpy数组以便后续处理

        # 计算CDF
        sorted_kv_diffs = np.sort(all_kv_diffs)  # 排序
        cdf = np.arange(1, len(sorted_kv_diffs) + 1) / len(sorted_kv_diffs)  # 计算CDF

        # 为每一层分配一个渐变色
        color = cmap(norm(i))  # 获取每层的颜色

        # 绘制当前层的CDF曲线
        ax.plot(sorted_kv_diffs, cdf, marker='.', linestyle='-', color=color)

    # 设置图表标题与标签
    ax.set_title(title or 'CDF of KV Diff for All Layers')
    ax.set_xlabel('KV Diff Value')
    ax.set_ylabel('CDF')
    ax.grid(True)

    # 绘制虚线（如果传入了阈值）
    if threshold is not None:
        ax.axvline(
            threshold, 
            color='red',  # 红色
            linestyle='--',  # 虚线
            linewidth=1.5,  # 线宽
            label=f'Threshold: {threshold}'  # 图例标签
        )
        ax.legend()  # 显示图例

    # 添加颜色条（用于显示渐变的图例）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 空数组，用于生成颜色条
    cbar = plt.colorbar(sm, ax=ax)  # 显式传递ax给colorbar
    cbar.set_label('Layer Number')  # 设置颜色条的标签

    # 保存图
    output_path = os.path.join(output_dir, f"{output_prefix}_kv_diff_cdf.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"CDF plot saved at {output_path}")

train_200_csv_file = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_train_max_200.csv'
test_200_csv_file = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_max_200.csv'
test_full_200_csv_file = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_max_200_full_200.csv'

if False and "compute kv diff":
    cache_1 = '/mnt/data/gbase/yinj/grkvc/cached_kv/model_ep50_15-02-02/ml_20m_sasrec_format_by_user_test_max_200_full_200_max_100/batch_0_cache_on_cpu.pt'
    cache_2 = '/mnt/data/gbase/yinj/grkvc/cached_kv/model_b50_15-02-16_ep3/ml_20m_sasrec_format_by_user_test_max_200_full_200_max_100/batch_0_cache_on_cpu.pt'
    length_file = '/mnt/data/gbase/yinj/grkvc/cached_kv/model_ep50_15-02-02/ml_20m_sasrec_format_by_user_test_max_200_full_200_max_100/batch_0_lengths_on_cpu.pt'
    output_dir = '/home/yinj@/datas/grkvc/diff_kv/'
    output_prefix = 'model_ep50_15-02-02_vs_model_b50_15-02-16_ep3'
    compute_kv_diff(file1=cache_1, file2=cache_2, length_file=length_file, output_dir=output_dir, output_prefix=output_prefix)

if False and "see the kv diff's content":
    k_diff_file = '/home/yinj@/datas/grkvc/diff_kv/model_ep50_15-02-02_vs_model_b50_15-02-16_ep3_k_diff.pt'
    kv_diff_file = '/home/yinj@/datas/grkvc/diff_kv/model_ep50_15-02-02_vs_model_b50_15-02-16_ep3_kv_diff.pt'
    k_diff = torch.load(k_diff_file)
    kv_diff = torch.load(kv_diff_file)
    print(f"k: len-{len(k_diff)}, shape-{k_diff[0].shape}, kv: len-{len(kv_diff)}, shape-{kv_diff[0].shape}")

if False and "[maybe wrong] plot kv diff heatmap by user_id and layer_id":
    # don't know why layer 0 has so significant diff, maybe because different model has different embedding table, which influence much more
    data_file_prefix = '/home/yinj@/datas/grkvc/diff_kv/15-02-02_vs_15-02-16_batch_1_'
    output_dir = '/home/yinj@/datas/grkvc/diff_kv/'
    output_prefix = '15-02-02_vs_15-02-16_batch_1_'
    suffixes = ['k_diff.pt', 'v_diff.pt', 'kv_diff.pt']
    token_start = 99
    token_end = 197
    for suffix in suffixes:
        plot_diff_heatmap(diff_file_path=data_file_prefix+suffix, output_dir=output_dir, output_prefix=output_prefix+suffix.replace('.pt', ''), token_start=token_start, token_end=token_end)

if False and "plot correlation":
    data_file_prefix = '/home/yinj@/datas/grkvc/diff_kv/model_ep50_15-02-02_vs_model_b50_15-02-16_ep3_'
    output_dir = '/home/yinj@/datas/grkvc/diff_kv/'
    output_prefix = 'model_ep50_15-02-02_vs_model_b50_15-02-16_ep3_'
    suffixes = ['k_diff.pt', 'v_diff.pt', 'kv_diff.pt']
    threshold = 0.75
    for suffix in suffixes:
        plot_layer_correlation(diff_file_path=data_file_prefix+suffix, output_dir=output_dir, output_prefix=output_prefix+suffix.replace('.pt', ''), threshold=threshold)

if False and "plot CDF":
    data_file_prefix = '/home/yinj@/datas/grkvc/diff_kv/model_ep50_15-02-02_vs_model_b50_15-02-16_ep3_'
    output_dir = '/home/yinj@/datas/grkvc/diff_kv/'
    output_prefix = 'model_ep50_15-02-02_vs_model_b50_15-02-16_ep3_'
    suffixes = ['kv_diff.pt']
    threshold = 0.75
    for suffix in suffixes:
        plot_kv_diff_cdf(diff_file_path=data_file_prefix+suffix, output_dir=output_dir, output_prefix=output_prefix+suffix.replace('.pt', ''))    

if False and "load cached_kv see what is in it":
    cached_kv_file = '/mnt/data/gbase/yinj/grkvc/cached_kv/15-02-16/ml_20m_sasrec_format_by_user_test_max_200_full_200_max_100/batch_1_cache_on_cpu.pt'
    data = torch.load(cached_kv_file)
    print(f"the len of data is {len(data)}, cached_v is {data[0][0].shape}, cached_k is {data[0][2].shape}")

if False and "process data for fullly seq_len ":
    filter_rows_by_seq_len(test_200_csv_file, max_seq_len=200)

if False and "convert log to csv":
    # log_file = '/home/yinj@/datas/grkvc/result_logs/see_delta_2_week_5_times.log'
    log_file = '/home/yinj@/datas/grkvc/perfect_datas/see_5_model_metrics_on_max_200.log'
    log_file = '/home/yinj@/datas/grkvc/perfect_datas/about_5_model_no_fully_selective_range_r_metrics_full_200.log'
    log_file = '/home/yinj@/datas/grkvc/perfect_datas/b150_d10_f200.log'
    log_file = '/home/yinj@/datas/grkvc/perfect_datas/b100_d5_f200.log'
    log_file = '/home/yinj@/datas/grkvc/result_logs/see_base_ep50_d3_3_type_range_r.log'
    log_file = '/home/yinj@/datas/grkvc/result_logs/see_base_ep50_d3_3_type_range_r_w_l2_distance.log'
    convert_log_to_csv(input_file=log_file)

if True and "plot recompute_ratio_metrics, recompute_ratio_time, loss pictures":
    csv_file = '/home/yinj@/datas/grkvc/perfect_datas/about_5_model_no_fully_selective_range_r_metrics_full_200.csv'
    csv_file = '/home/yinj@/datas/grkvc/perfect_datas/b150_d10_f200.csv'
    csv_file = '/home/yinj@/datas/grkvc/perfect_datas/b100_d5_f200.csv'
    csv_file = '/home/yinj@/datas/grkvc/result_logs/see_base_ep50_d3_3_type_range_r.csv'
    csv_file = '/home/yinj@/datas/grkvc/result_logs/see_base_ep50_d3_3_type_range_r_w_l2_distance.csv'
    selected_metrics = ['NDCG@10', 'HR@10', 'MRR']
    skip_model = 'model_ep20_15-02-02'
    skip_model = 'model_ep50_15-02-02'
    image_prefix = 'b100_d10'
    image_prefix = 'b100_d10_l2'
    # image_prefix = 'b150_d10'
    # image_prefix = 'b100_d5'
    y_lim = True
    plot_metrics(csv_file=csv_file, selected_metrics=selected_metrics, skip_model=skip_model, image_prefix=image_prefix, y_lim=y_lim)

if False and "plot one model selective metrics from full test csv":
    csv_file = '/home/yinj@/datas/grkvc/perfect_datas/about_5_model_no_fully_selective_range_r_metrics_full_200.csv'
    model_name = 'model_ep20_15-02-16'
    plot_1_model_selective_metrics_from_full_test_csv(csv_file=csv_file, model_name=model_name)

if False and "plot metrics from csv":
    # csv_file = '/home/yinj@/datas/grkvc/result_logs/see_delta_2_week_5_times.csv'
    csv_file = '/home/yinj@/datas/grkvc/perfect_datas/see_5_model_metrics_on_max_200.csv'
    output_dir = '/home/yinj@/datas/grkvc/graphs'
    # graph_name = 'tmp_see_all_metrics.png'
    graph_name = 'see_5_model_metrics_on_max_200.png'
    plot_metrics_from_csv(csv_file=csv_file, output_dir=output_dir, output_name=graph_name)

if False and "plot selective metrics from csv":
    csv_file = '/home/yinj@/datas/grkvc/perfect_datas/see_5_model_metrics_on_max_200.csv'
    output_dir = '/home/yinj@/datas/grkvc/graphs'
    graph_name = 'see_5_model_metrics_on_max_200.png'
    plot_selective_metrics_from_csv(csv_file=csv_file, output_dir=output_dir, output_name=graph_name)

if False and "parse log then plot selective metrics":
    log_file = '/home/yinj@/datas/grkvc/perfect_datas/see_5_model_metrics_on_max_200_max_100.log'
    output_dir = '/home/yinj@/datas/grkvc/graphs'
    csv_file = convert_log_to_csv(input_file=log_file)
    graph_name = os.path.basename(log_file).replace('.log', '.png')
    plot_selective_metrics_from_csv(csv_file=csv_file, output_dir=output_dir, output_name=graph_name)

if False and "truncate data by max_len":
    inputfile = test_full_200_csv_file
    # truncate_sequence_data(inputfile=inputfile, max_seq_len=50)
    # truncate_sequence_data(inputfile=inputfile, max_seq_len=60)
    # truncate_sequence_data(inputfile=inputfile, max_seq_len=100)
    # truncate_sequence_data(inputfile=inputfile, max_seq_len=110)
    # truncate_sequence_data(inputfile=inputfile, max_seq_len=150)
    # truncate_sequence_data(inputfile=inputfile, max_seq_len=160)
    truncate_sequence_data(inputfile=inputfile, max_seq_len=105)
    truncate_sequence_data(inputfile=inputfile, max_seq_len=120)
    truncate_sequence_data(inputfile=inputfile, max_seq_len=155)
    truncate_sequence_data(inputfile=inputfile, max_seq_len=170)

if False and "truncate data by timestamp":
    dates = ["2015-02-03", "2015-02-17", "2015-03-03", "2015-03-17", "2015-03-31"]
    process_data_by_timestamps(test_200_csv_file, dates=dates)

if False and "extract metrics from log":
    log_file = ''
    output_file = ''
    extract_metrics_from_log(log_file_path=log_file, output_csv_path=output_file)