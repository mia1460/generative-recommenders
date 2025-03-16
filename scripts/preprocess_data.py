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
        pattern = r'\[loading checkpoint\] from: (.*?)\n.*?metrics are: (.*?)\n'
        matches = re.findall(pattern, log_content, re.DOTALL)

        data = []
        for match in matches:
            ckpt_path = match[0].strip()
            ckpt_file = os.path.basename(ckpt_path)
            model_name = os.path.splitext(ckpt_file)[0]
            metrics_str = match[1].strip()

            metrics = {}
            metrics_pairs = re.findall(r'([A-Z]+@\d+|\bMRR\b)\s+([0-9.]+)', metrics_str)
            for key, value in metrics_pairs:
                metrics[key] = float(value)

            data.append({
                'cache_type': 'no',
                'model_name': model_name,
                **metrics
            })

        return data

    metrics_data = extract_metrics(content)
    
    if not metrics_data:
        print("No valid data found.")
        return
    
    fieldnames = ['cache_type', 'model_name'] + list(metrics_data[0].keys())[2:]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_data)
    print(f"metrics saved at {output_file}")

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

train_200_csv_file = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_train_max_200.csv'
test_200_csv_file = '/home/yinj@/datas/grkvc/use_data/ml_20m_sasrec_format_by_user_test_max_200.csv'

if False and "convert log to csv":
    # log_file = '/home/yinj@/datas/grkvc/result_logs/see_delta_2_week_5_times.log'
    log_file = '/home/yinj@/datas/grkvc/perfect_datas/see_5_model_metrics_on_max_200.log'
    convert_log_to_csv(input_file=log_file)

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

if True and "parse log then plot selective metrics":
    log_file = '/home/yinj@/datas/grkvc/perfect_datas/see_5_model_metrics_on_max_200_max_100.log'
    output_dir = '/home/yinj@/datas/grkvc/graphs'
    csv_file = convert_log_to_csv(input_file=log_file)
    graph_name = os.path.basename(log_file).replace('.log', '.png')
    plot_selective_metrics_from_csv(csv_file=csv_file, output_dir=output_dir, output_name=graph_name)

if False and "truncate data by max_len":
    inputfile = test_200_csv_file
    truncate_sequence_data(inputfile=inputfile, max_seq_len=50)
    truncate_sequence_data(inputfile=inputfile, max_seq_len=60)
    truncate_sequence_data(inputfile=inputfile, max_seq_len=100)
    truncate_sequence_data(inputfile=inputfile, max_seq_len=110)
    truncate_sequence_data(inputfile=inputfile, max_seq_len=150)
    truncate_sequence_data(inputfile=inputfile, max_seq_len=160)

if False and "truncate data by timestamp":
    dates = ["2015-02-03", "2015-02-17", "2015-03-03", "2015-03-17", "2015-03-31"]
    process_data_by_timestamps(test_200_csv_file, dates=dates)

if False and "extract metrics from log":
    log_file = ''
    output_file = ''
    extract_metrics_from_log(log_file_path=log_file, output_csv_path=output_file)