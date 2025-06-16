#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA微调实验完整版

本脚本包含完整的LoRA微调实验，包括：
1. 基础训练和loss曲线绘制
2. LoRA超参数调试
3. Alpaca数据集训练
4. 两个数据集的超参数对比分析
"""

import os
import warnings
import json
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, PeftModel

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 加载模型和tokenizer
model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
foundation_model = AutoModelForCausalLM.from_pretrained(model_name)

# 如果tokenizer没有pad_token，使用eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"模型加载完成: {model_name}")
print(f"模型参数量: {foundation_model.num_parameters():,}")




def load_model_and_tokenizer(model_name="bigscience/bloomz-560m"):
    """加载模型和tokenizer"""
    print(f"加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    foundation_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 如果tokenizer没有pad_token，使用eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"模型加载完成: {model_name}")
    print(f"模型参数量: {foundation_model.num_parameters():,}")
    
    return foundation_model, tokenizer

def get_outputs(model, tokenized_inputs, max_new_tokens=100):
    """生成文本的函数"""
    generated_outputs = model.generate(
        input_ids=tokenized_inputs["input_ids"].to(model.device),
        attention_mask=tokenized_inputs["attention_mask"].to(model.device),
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7
    )
    return generated_outputs

def test_model_generation(model, test_prompt="I love this movie because", tokenizer=None):
    """测试模型生成效果"""
    if tokenizer is None:
        raise ValueError("tokenizer不能为None")
    
    tokenized_input = tokenizer(test_prompt, return_tensors="pt")
    generated_outputs = get_outputs(model, tokenized_input, max_new_tokens=50)
    generated_text = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)[0]
    return generated_text

def load_imdb_dataset(tokenizer, num_samples=50):
    """加载IMDB数据集"""
    print("加载IMDB数据集...")
    
        
    #Create the Dataset to create prompts.
    raw_dataset = load_dataset("noob123/imdb_review_3000")
    tokenized_dataset = raw_dataset.map(lambda samples: tokenizer(samples['review']), batched=True)
    training_subset = tokenized_dataset["train"].select(range(num_samples))

    training_subset = training_subset.remove_columns('sentiment')
        
    return training_subset

def load_alpaca_dataset(tokenizer, num_samples=100):
    """加载Alpaca数据集"""
    print("加载Alpaca数据集...")
    try:
        raw_alpaca_dataset = load_dataset("tatsu-lab/alpaca")
        
        def format_alpaca_prompt(examples):
            """格式化Alpaca提示"""
            formatted_prompts = []
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                input_text = examples['input'][i] if examples['input'][i] else ""
                output = examples['output'][i]
                
                if input_text:
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                formatted_prompts.append(prompt)
            
            return {'text': formatted_prompts}
        
        formatted_dataset = raw_alpaca_dataset.map(format_alpaca_prompt, batched=True)
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)
        
        tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)
        training_subset = tokenized_dataset["train"].select(range(num_samples))
        
        # 只保留必要的列
        training_subset = training_subset.remove_columns(['instruction', 'input', 'output', 'text'])
        
        return training_subset
        
    except Exception as e:
        print(f"加载Alpaca数据集失败: {e}")
        print("使用备用数据集...")
        # 创建一个简单的指令数据集作为备用
        fallback_instruction_data = [
            "### Instruction:\nWrite a short story about a robot.\n\n### Response:\nOnce upon a time, there was a kind robot named Zara who helped people in the city.",
            "### Instruction:\nExplain what machine learning is.\n\n### Response:\nMachine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "### Instruction:\nWrite a poem about nature.\n\n### Response:\nTrees sway gently in the breeze, Birds sing melodies with ease.",
        ] * (num_samples // 3 + 1)
        
        fallback_instruction_data = fallback_instruction_data[:num_samples]
        
        fallback_dataset = Dataset.from_dict({'text': fallback_instruction_data})
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)
        
        tokenized_fallback_dataset = fallback_dataset.map(tokenize_function, batched=True)
        tokenized_fallback_dataset = tokenized_fallback_dataset.remove_columns(['text'])
        
        return tokenized_fallback_dataset

class CustomTrainer(Trainer):
    """自定义Trainer类，用于记录详细的训练损失"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history = []
        self.step_history = []
    
    def log(self, logs, start_time=None):
        # 兼容不同版本的transformers
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
            
        if 'loss' in logs:
            self.loss_history.append(logs['loss'])
            self.step_history.append(self.state.global_step)

def plot_training_loss(trainer, title="Training Loss", output_dir="./"):
    """绘制训练损失曲线"""
    if hasattr(trainer, 'loss_history') and trainer.loss_history:
        plt.figure(figsize=(10, 6))
        plt.plot(trainer.step_history, trainer.loss_history, 'b-', linewidth=2, markersize=4)
        plt.title(title, fontsize=16)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "imdb_first_train.png"))
        plt.show()
        
        print(f"初始损失: {trainer.loss_history[0]:.4f}")
        print(f"最终损失: {trainer.loss_history[-1]:.4f}")
        print(f"损失下降: {(trainer.loss_history[0] - trainer.loss_history[-1]):.4f}")
    else:
        print("没有找到损失历史记录")

def train_lora_with_params(rank, alpha, dropout, dataset, dataset_name, model_name, epochs=2):
    """使用指定参数训练LoRA模型"""
    
    print(f"\n=== 训练LoRA模型 ===")
    print(f"参数: r={rank}, alpha={alpha}, dropout={dropout}")
    print(f"数据集: {dataset_name}")
    
    # 创建新的基础模型实例
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 配置LoRA
    lora_hyperparameters = LoraConfig(
        r=int(rank),
        lora_alpha=float(alpha),
        target_modules=["query_key_value"],
        lora_dropout=float(dropout),
        bias="lora_only",
        task_type="CAUSAL_LM"
    )
    
    # 创建LoRA模型
    peft_model = get_peft_model(base_model, lora_hyperparameters)
    
    # 设置输出目录
    output_dir = f"./lora_r{rank}_a{alpha}_d{dropout}_{dataset_name}"
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=3e-4,  # 降低学习率，避免梯度爆炸
        num_train_epochs=2,
        logging_steps=5,  # 更频繁的日志记录
        save_steps=100,
        #evaluation_strategy="no",
        use_cpu=False,
        #max_grad_norm=1.0,  # 添加梯度裁剪
        #warmup_steps=10,    # 添加学习率预热
    )
    
    # 创建训练器
    trainer = CustomTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 训练
    training_result = trainer.train()
    
    # 保存模型
    model_save_path = os.path.join(output_dir, "lora_model")
    trainer.model.save_pretrained(model_save_path)
    
    # 计算可训练参数
    trainable_param_count = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_param_count = sum(p.numel() for p in peft_model.parameters())
    
    evaluation_model = PeftModel.from_pretrained(base_model, model_save_path, is_trainable=False)
    evaluation_results = []
    evaluation_prompts = [
        "I love this movie because",
        "This film is amazing and",
        "The story was boring but",
        "Great acting and",
        "The plot is interesting since",
        "I hate this movie because",
        "The characters are well developed and"
    ]
    for i, prompt in enumerate(evaluation_prompts, 1):
        print(f"测试样例 {i}: {prompt}")
        
        # 生成文本
        tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        generated_outputs = get_outputs(evaluation_model, tokenized_prompt, max_new_tokens=50)
        decoded_text = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)[0]
        
        # 保存结果
        evaluation_results.append(f"\n{i}. 输入提示: {prompt}"+f"\n生成结果: {decoded_text}")
       
    
    # 保存测试结果到文件
    evaluation_results_file = os.path.join(output_dir, "test_generation_results.txt")
    with open(evaluation_results_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(evaluation_results))
    
    print(f"\n测试结果已保存到: {evaluation_results_file}")
    
    plot_training_loss(trainer, title="Training Loss", output_dir=output_dir)
    
    # 显式清理GPU内存
    import gc
    import torch
    
    # 删除模型对象
    del evaluation_model
    del peft_model
    del base_model
    #del trainer
    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"GPU内存已清理")
    
    return {
        'rank': rank,
        'alpha': alpha,
        'dropout': dropout,
        'dataset': dataset_name,
        'final_loss': training_result.training_loss,
        'trainable_params': trainable_param_count,
        'total_params': total_param_count,
        'train_time': training_result.metrics['train_runtime'],
        'trainer': trainer,
        'model_path': model_save_path
    }

def analyze_hyperparameter_results(experiment_results, dataset_name):
    """分析超参数实验结果"""
    
    if not experiment_results:
        print("没有实验结果可以分析")
        return None
    
    print(f"\n=== {dataset_name}数据集超参数分析 ===")
    
    # 创建结果DataFrame
    results_dataframe = pd.DataFrame(experiment_results)
    
    # 显示结果表格
    print("\n实验结果汇总:")
    display_columns = ['rank', 'alpha', 'dropout', 'final_loss', 'trainable_params', 'train_time']
    formatted_result_table = results_dataframe[display_columns].copy()
    formatted_result_table['trainable_params'] = formatted_result_table['trainable_params'].apply(lambda x: f"{x:,}")
    formatted_result_table['train_time'] = formatted_result_table['train_time'].apply(lambda x: f"{x:.1f}s")
    formatted_result_table['final_loss'] = formatted_result_table['final_loss'].apply(lambda x: f"{x:.4f}")
    print(formatted_result_table.to_string(index=False))
    
    # 找到最佳配置
    best_config_index = results_dataframe['final_loss'].idxmin()
    best_configuration = results_dataframe.iloc[best_config_index]
    print(f"\n最佳配置:")
    print(f"  Rank: {best_configuration['rank']}")
    print(f"  Alpha: {best_configuration['alpha']}")
    print(f"  Dropout: {best_configuration['dropout']}")
    print(f"  最终损失: {best_configuration['final_loss']:.4f}")
    print(f"  可训练参数: {best_configuration['trainable_params']:,}")
    
    # 绘制分析图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 损失对比
    loss_comparison_axis = axes[0, 0]
    hyperparameter_labels = [f"r{r}_a{a}_d{d}" for r, a, d in zip(results_dataframe['rank'], results_dataframe['alpha'], results_dataframe['dropout'])]
    loss_comparison_bars = loss_comparison_axis.bar(range(len(results_dataframe)), results_dataframe['final_loss'], color='skyblue', alpha=0.7)
    loss_comparison_axis.set_xlabel('配置')
    loss_comparison_axis.set_ylabel('最终损失')
    loss_comparison_axis.set_title('不同配置的最终损失对比')
    loss_comparison_axis.set_xticks(range(len(results_dataframe)))
    loss_comparison_axis.set_xticklabels(hyperparameter_labels, rotation=45, ha='right')
    
    # 标注最低点
    min_loss_index = results_dataframe['final_loss'].idxmin()
    loss_comparison_axis.annotate(f'最佳: {results_dataframe.iloc[min_loss_index]["final_loss"]:.4f}', 
                xy=(min_loss_index, results_dataframe.iloc[min_loss_index]['final_loss']),
                xytext=(min_loss_index, results_dataframe.iloc[min_loss_index]['final_loss'] + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red', fontweight='bold')
    
    # 2. 参数量对比
    parameter_comparison_axis = axes[0, 1]
    parameter_comparison_bars = parameter_comparison_axis.bar(range(len(results_dataframe)), results_dataframe['trainable_params'], color='lightgreen', alpha=0.7)
    parameter_comparison_axis.set_xlabel('配置')
    parameter_comparison_axis.set_ylabel('可训练参数数量')
    parameter_comparison_axis.set_title('不同配置的参数量对比')
    parameter_comparison_axis.set_xticks(range(len(results_dataframe)))
    parameter_comparison_axis.set_xticklabels(hyperparameter_labels, rotation=45, ha='right')
    
    # 3. Rank影响分析
    rank_analysis_axis = axes[1, 0]
    # 筛选相同alpha和dropout的配置
    rank_baseline_configs = results_dataframe[(results_dataframe['alpha'] == 8) & (results_dataframe['dropout'] == 0.05)]
    if len(rank_baseline_configs) > 1:
        rank_analysis_axis.plot(rank_baseline_configs['rank'], rank_baseline_configs['final_loss'], 'o-', linewidth=2, markersize=8)
        rank_analysis_axis.set_xlabel('Rank (r)')
        rank_analysis_axis.set_ylabel('最终损失')
        rank_analysis_axis.set_title('Rank对损失的影响 (alpha=8, dropout=0.05)')
        rank_analysis_axis.grid(True, alpha=0.3)
    else:
        rank_analysis_axis.text(0.5, 0.5, 'Rank分析数据不足', ha='center', va='center', transform=rank_analysis_axis.transAxes)
    
    # 4. Alpha影响分析
    alpha_analysis_axis = axes[1, 1]
    alpha_baseline_configs = results_dataframe[(results_dataframe['rank'] == 4) & (results_dataframe['dropout'] == 0.05)]
    if len(alpha_baseline_configs) > 1:
        alpha_analysis_axis.plot(alpha_baseline_configs['alpha'], alpha_baseline_configs['final_loss'], 's-', linewidth=2, markersize=8, color='orange')
        alpha_analysis_axis.set_xlabel('Alpha')
        alpha_analysis_axis.set_ylabel('最终损失')
        alpha_analysis_axis.set_title('Alpha对损失的影响 (rank=4, dropout=0.05)')
        alpha_analysis_axis.grid(True, alpha=0.3)
    else:
        alpha_analysis_axis.text(0.5, 0.5, 'Alpha分析数据不足', ha='center', va='center', transform=alpha_analysis_axis.transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_hyperparameter_analysis.png")
    plt.show()
    
    # 分析结论
    print(f"\n=== 分析结论 ===")
    
    # Rank分析
    rank_analysis_configs = results_dataframe[(results_dataframe['alpha'] == 8) & (results_dataframe['dropout'] == 0.05)].sort_values('rank')
    if len(rank_analysis_configs) > 1:
        print("Rank影响:")
        print(f"  - 最低Rank ({rank_analysis_configs.iloc[0]['rank']}): 损失 {rank_analysis_configs.iloc[0]['final_loss']:.4f}")
        print(f"  - 最高Rank ({rank_analysis_configs.iloc[-1]['rank']}): 损失 {rank_analysis_configs.iloc[-1]['final_loss']:.4f}")
        if rank_analysis_configs.iloc[0]['final_loss'] > rank_analysis_configs.iloc[-1]['final_loss']:
            print("  - 结论: 高Rank表现更好，但参数量增加")
        else:
            print("  - 结论: 低Rank已足够，更高效")
    
    # Alpha分析
    alpha_analysis_configs = results_dataframe[(results_dataframe['rank'] == 4) & (results_dataframe['dropout'] == 0.05)].sort_values('alpha')
    if len(alpha_analysis_configs) > 1:
        print("Alpha影响:")
        best_alpha_index = alpha_analysis_configs['final_loss'].idxmin()
        best_alpha_configuration = alpha_analysis_configs.loc[best_alpha_index]
        print(f"  - 最佳Alpha: {best_alpha_configuration['alpha']}, 损失: {best_alpha_configuration['final_loss']:.4f}")
    
    # Dropout分析
    dropout_analysis_configs = results_dataframe[(results_dataframe['rank'] == 4) & (results_dataframe['alpha'] == 8)].sort_values('dropout')
    if len(dropout_analysis_configs) > 1:
        print("Dropout影响:")
        best_dropout_index = dropout_analysis_configs['final_loss'].idxmin()
        best_dropout_configuration = dropout_analysis_configs.loc[best_dropout_index]
        print(f"  - 最佳Dropout: {best_dropout_configuration['dropout']}, 损失: {best_dropout_configuration['final_loss']:.4f}")
    
    return best_configuration

def compare_datasets_hyperparameters(imdb_results, alpaca_results, imdb_best, alpaca_best):
    """对比两个数据集的超参数结果"""
    
    # 创建对比DataFrame
    imdb_df = pd.DataFrame(imdb_results)
    alpaca_df = pd.DataFrame(alpaca_results)
    
    # 添加数据集标识
    imdb_df['dataset'] = 'IMDB'
    alpaca_df['dataset'] = 'Alpaca'
    
    # 合并数据
    combined_df = pd.concat([imdb_df, alpaca_df], ignore_index=True)
    
    # 绘制对比图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 损失对比 - 按数据集分组
    ax1 = axes[0, 0]
    imdb_losses = imdb_df['final_loss'].values
    alpaca_losses = alpaca_df['final_loss'].values
    
    x_pos = np.arange(len(imdb_losses))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, imdb_losses, width, label='IMDB', alpha=0.7, color='skyblue')
    bars2 = ax1.bar(x_pos + width/2, alpaca_losses, width, label='Alpaca', alpha=0.7, color='lightcoral')
    
    ax1.set_xlabel('配置索引')
    ax1.set_ylabel('最终损失')
    ax1.set_title('两个数据集的损失对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 标注最佳点
    imdb_best_idx = imdb_df['final_loss'].idxmin()
    alpaca_best_idx = alpaca_df['final_loss'].idxmin()
    ax1.annotate(f'IMDB最佳', xy=(imdb_best_idx - width/2, imdb_losses[imdb_best_idx]),
                xytext=(imdb_best_idx - width/2, imdb_losses[imdb_best_idx] + 0.1),
                arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
    ax1.annotate(f'Alpaca最佳', xy=(alpaca_best_idx + width/2, alpaca_losses[alpaca_best_idx]),
                xytext=(alpaca_best_idx + width/2, alpaca_losses[alpaca_best_idx] + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'), color='red')
    
    # 2. Rank对损失的影响对比
    ax2 = axes[0, 1]
    for dataset, df in [('IMDB', imdb_df), ('Alpaca', alpaca_df)]:
        rank_configs = df[(df['alpha'] == 8) & (df['dropout'] == 0.05)].sort_values('rank')
        if len(rank_configs) > 1:
            ax2.plot(rank_configs['rank'], rank_configs['final_loss'], 'o-', 
                    linewidth=2, markersize=8, label=f'{dataset}')
    ax2.set_xlabel('Rank (r)')
    ax2.set_ylabel('最终损失')
    ax2.set_title('Rank对损失的影响对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Alpha对损失的影响对比
    ax3 = axes[0, 2]
    for dataset, df in [('IMDB', imdb_df), ('Alpaca', alpaca_df)]:
        alpha_configs = df[(df['rank'] == 4) & (df['dropout'] == 0.05)].sort_values('alpha')
        if len(alpha_configs) > 1:
            ax3.plot(alpha_configs['alpha'], alpha_configs['final_loss'], 's-', 
                    linewidth=2, markersize=8, label=f'{dataset}')
    ax3.set_xlabel('Alpha')
    ax3.set_ylabel('最终损失')
    ax3.set_title('Alpha对损失的影响对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Dropout对损失的影响对比
    ax4 = axes[1, 0]
    for dataset, df in [('IMDB', imdb_df), ('Alpaca', alpaca_df)]:
        dropout_configs = df[(df['rank'] == 4) & (df['alpha'] == 8)].sort_values('dropout')
        if len(dropout_configs) > 1:
            ax4.plot(dropout_configs['dropout'], dropout_configs['final_loss'], '^-', 
                    linewidth=2, markersize=8, label=f'{dataset}')
    ax4.set_xlabel('Dropout')
    ax4.set_ylabel('最终损失')
    ax4.set_title('Dropout对损失的影响对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 参数量对比
    ax5 = axes[1, 1]
    imdb_params = imdb_df['trainable_params'].values
    alpaca_params = alpaca_df['trainable_params'].values
    
    bars3 = ax5.bar(x_pos - width/2, imdb_params, width, label='IMDB', alpha=0.7, color='lightgreen')
    bars4 = ax5.bar(x_pos + width/2, alpaca_params, width, label='Alpaca', alpha=0.7, color='orange')
    
    ax5.set_xlabel('配置索引')
    ax5.set_ylabel('可训练参数数量')
    ax5.set_title('可训练参数数量对比')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 最佳配置对比雷达图
    ax6 = axes[1, 2]
    
    # 准备雷达图数据
    categories = ['Rank', 'Alpha', 'Dropout', 'Loss (归一化)']
    
    # 归一化损失值以便于比较
    max_loss = max(imdb_best['final_loss'], alpaca_best['final_loss'])
    min_loss = min(imdb_best['final_loss'], alpaca_best['final_loss'])
    
    imdb_values = [
        imdb_best['rank'],
        imdb_best['alpha'] / 32,  # 归一化到0-1
        imdb_best['dropout'] * 5,  # 放大以便可视化
        1 - (imdb_best['final_loss'] - min_loss) / (max_loss - min_loss) if max_loss != min_loss else 0.5
    ]
    
    alpaca_values = [
        alpaca_best['rank'],
        alpaca_best['alpha'] / 32,
        alpaca_best['dropout'] * 5,
        1 - (alpaca_best['final_loss'] - min_loss) / (max_loss - min_loss) if max_loss != min_loss else 0.5
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    imdb_values += imdb_values[:1]
    alpaca_values += alpaca_values[:1]
    
    ax6.plot(angles, imdb_values, 'o-', linewidth=2, label='IMDB', color='blue')
    ax6.fill(angles, imdb_values, alpha=0.25, color='blue')
    ax6.plot(angles, alpaca_values, 's-', linewidth=2, label='Alpaca', color='red')
    ax6.fill(angles, alpaca_values, alpha=0.25, color='red')
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, max(max(imdb_values), max(alpaca_values)) * 1.1)
    ax6.set_title('最佳配置对比 (雷达图)')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig("compare_hyperparameter.png")
    plt.show()

def analyze_hyperparameter_differences(imdb_best, alpaca_best, imdb_results, alpaca_results):
    """深入分析两个数据集超参数差异的原因"""
    
    print("基于数据集特性的差异分析:")
    
    # 1. 数据集特性分析
    print("\n1. 数据集特性差异:")
    print("   IMDB数据集:")
    print("   - 任务类型: 情感分析/文本生成")
    print("   - 数据特点: 电影评论，情感表达丰富")
    print("   - 序列长度: 中等长度，情感词汇集中")
    print("   - 学习目标: 学习情感表达模式")
    
    print("\n   Alpaca数据集:")
    print("   - 任务类型: 指令跟随")
    print("   - 数据特点: 结构化指令-响应对")
    print("   - 序列长度: 变化较大，包含长篇回答")
    print("   - 学习目标: 学习指令理解和多样化响应")
    
    # 2. 超参数需求分析
    print("\n2. 超参数需求差异分析:")
    
    rank_diff = alpaca_best['rank'] - imdb_best['rank']
    alpha_diff = alpaca_best['alpha'] - imdb_best['alpha']
    dropout_diff = alpaca_best['dropout'] - imdb_best['dropout']
    
    if rank_diff > 0:
        print(f"   Rank: Alpaca需要更高的rank ({alpaca_best['rank']} vs {imdb_best['rank']})")
        print("   原因推测: 指令跟随任务更复杂，需要更多的低秩矩阵来捕获多样化的指令-响应模式")
    elif rank_diff < 0:
        print(f"   Rank: IMDB需要更高的rank ({imdb_best['rank']} vs {alpaca_best['rank']})")
        print("   原因推测: 情感分析可能需要更复杂的特征提取")
    else:
        print(f"   Rank: 两个数据集最佳rank相同 ({imdb_best['rank']})")
        print("   说明: 两个任务对模型容量的需求相似")
    
    if alpha_diff > 0:
        print(f"   Alpha: Alpaca需要更高的alpha ({alpaca_best['alpha']} vs {imdb_best['alpha']})")
        print("   原因推测: 指令跟随需要更强的LoRA影响力来覆盖原始权重")
    elif alpha_diff < 0:
        print(f"   Alpha: IMDB需要更高的alpha ({imdb_best['alpha']} vs {alpaca_best['alpha']})")
        print("   原因推测: 情感分析需要更强的权重调整")
    else:
        print(f"   Alpha: 两个数据集最佳alpha相同 ({imdb_best['alpha']})")
    
    if dropout_diff > 0.01:
        print(f"   Dropout: Alpaca需要更高的dropout ({alpaca_best['dropout']:.2f} vs {imdb_best['dropout']:.2f})")
        print("   原因推测: 指令数据更多样化，需要更强的正则化防止过拟合")
    elif dropout_diff < -0.01:
        print(f"   Dropout: IMDB需要更高的dropout ({imdb_best['dropout']:.2f} vs {alpaca_best['dropout']:.2f})")
        print("   原因推测: 情感数据可能存在过拟合风险")
    else:
        print(f"   Dropout: 两个数据集最佳dropout相近 ({imdb_best['dropout']:.2f})")
    
    # 3. 性能分析
    print("\n3. 训练效果分析:")
    loss_diff = alpaca_best['final_loss'] - imdb_best['final_loss']
    if loss_diff > 0.1:
        print(f"   Alpaca的损失较高 ({alpaca_best['final_loss']:.4f} vs {imdb_best['final_loss']:.4f})")
        print("   可能原因: 指令跟随任务本身更复杂，需要学习更多样化的模式")
    elif loss_diff < -0.1:
        print(f"   IMDB的损失较高 ({imdb_best['final_loss']:.4f} vs {alpaca_best['final_loss']:.4f})")
        print("   可能原因: 情感分析任务对模型来说更具挑战性")
    else:
        print(f"   两个数据集的最终损失相近")
        print("   说明: LoRA在两个任务上都取得了相似的学习效果")
    
    # 4. 实际应用建议
    print("\n4. 实际应用建议:")
    print("   - 对于情感分析类任务:")
    print(f"     推荐参数: r={imdb_best['rank']}, alpha={imdb_best['alpha']}, dropout={imdb_best['dropout']:.2f}")
    print("   - 对于指令跟随类任务:")
    print(f"     推荐参数: r={alpaca_best['rank']}, alpha={alpaca_best['alpha']}, dropout={alpaca_best['dropout']:.2f}")
    print("   - 超参数选择应考虑:")
    print("     * 任务复杂度 (影响rank选择)")
    print("     * 数据多样性 (影响dropout选择)")
    print("     * 期望的权重调整强度 (影响alpha选择)")

def main():
    """主函数"""
    print("=== LoRA微调实验完整版 ===")
    
    # 加载模型和tokenizer
    model_name = "bigscience/bloomz-560m"
    foundation_model, tokenizer = load_model_and_tokenizer(model_name)
    
    # 测试原始模型生成效果
    print("\n=== 原始模型生成效果 ===")
    original_result = test_model_generation(foundation_model, tokenizer=tokenizer)
    print(f"原始模型输出:\n{original_result}\n")
    
    # 加载IMDB数据集
    print("加载IMDB数据集...")
    imdb_train_data = load_imdb_dataset(tokenizer, num_samples=100)
    print(f"IMDB数据集加载完成，样本数: {len(imdb_train_data)}")
    
    # 查看数据集样本
    print("\n=== IMDB数据集样本 ===")
    dataset_sample = imdb_train_data[0]
    print(f"输入长度: {len(dataset_sample['input_ids'])}")
    print(f"样本文本片段: {tokenizer.decode(dataset_sample['input_ids'][:100])}...")
    
    # 得分点2：使用原始数据集完成首次训练，提交训练loss曲线和模型生成结果
    print("\n=== 得分点2：基础LoRA训练 ===")
    
    # 配置基础LoRA参数
    basic_lora_configuration = LoraConfig(
        r=4,
        lora_alpha=1,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="lora_only",
        task_type="CAUSAL_LM"
    )
    
    # 创建LoRA模型
    print("创建LoRA模型...")
    basic_peft_model = get_peft_model(foundation_model, basic_lora_configuration)
    print(basic_peft_model.print_trainable_parameters())
    
    # 设置训练参数
    output_directory = "./peft_outputs_basic"
    training_args = TrainingArguments(
        output_dir=output_directory,
        auto_find_batch_size=True,
        learning_rate=3e-4,  # 降低学习率，避免梯度爆炸
        num_train_epochs=2,
        logging_steps=5,  # 更频繁的日志记录
        save_steps=100,
        #evaluation_strategy="no",
        use_cpu=False,
        #max_grad_norm=1.0,  # 添加梯度裁剪
        #warmup_steps=10,    # 添加学习率预热
        
    )
    
    print(f"训练参数设置完成，输出目录: {output_directory}")
    
    # 开始基础训练
    print("=== 开始基础LoRA训练 ===")
    
    trainer = CustomTrainer(
        model=basic_peft_model,
        args=training_args,
        train_dataset=imdb_train_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 训练模型
    train_result = trainer.train()
    
    # 绘制训练损失曲线
    print("\n=== 训练损失曲线 ===")
    plot_training_loss(trainer, "IMDB数据集基础LoRA训练损失")
    
    # 保存模型
    basic_model_path = os.path.join(output_directory, "basic_lora_model")
    trainer.model.save_pretrained(basic_model_path)
    print(f"模型已保存到: {basic_model_path}")
    
    # 显示训练统计信息
    print(f"\n=== 训练统计信息 ===")
    print(f"训练损失: {train_result.training_loss:.4f}")
    print(f"训练步数: {train_result.global_step}")
    print(f"训练时间: {train_result.metrics['train_runtime']:.2f}秒")
    
    # 测试训练后的模型生成效果
    print("=== 训练后模型生成效果对比 ===")
    
    # 加载训练后的模型
    trained_model = PeftModel.from_pretrained(foundation_model, basic_model_path, is_trainable=False)
    
    # 测试多个样本
    test_prompts = [
        "I love this movie because",
        "This film is amazing and",
        "The story was boring but",
        "Great acting and"
    ]
    
    print("原始模型 vs 训练后模型对比:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. 测试提示: '{prompt}'")
        
        # 原始模型输出
        original_output = test_model_generation(foundation_model, prompt, tokenizer)
        print(f"   原始模型: {original_output}")
        
        # 训练后模型输出
        trained_output = test_model_generation(trained_model, prompt, tokenizer)
        print(f"   训练后模型: {trained_output}")
    
    # 得分点3：LoRA超参数调试和分析
    print("\n=== 得分点3：LoRA超参数调试和分析 ===")
    
    # 定义超参数实验配置 - 更细致的搜索空间
    selected_configs = [
        # === Rank系列实验 (固定alpha=1, dropout=0.05) ===
        {'rank': 1, 'alpha': 1, 'dropout': 0.05},   # 极低rank
        {'rank': 2, 'alpha': 1, 'dropout': 0.05},   # 低rank
        {'rank': 4, 'alpha': 1, 'dropout': 0.05},   # 基准rank
        {'rank': 8, 'alpha': 1, 'dropout': 0.05},   # 中rank
        {'rank': 16, 'alpha': 1, 'dropout': 0.05},  # 高rank
  
        
        # === Alpha系列实验 (固定rank=4, dropout=0.05) ===
        {'rank': 4, 'alpha': 0.25, 'dropout': 0.05},  # 极低alpha
        {'rank': 4, 'alpha': 0.5, 'dropout': 0.05},   # 很低alpha
        {'rank': 4, 'alpha': 1, 'dropout': 0.05},     # 基准alpha (重复，用作参考)
        {'rank': 4, 'alpha': 2, 'dropout': 0.05},     # 低alpha
        {'rank': 4, 'alpha': 4, 'dropout': 0.05},     # 中alpha (alpha=rank)
        {'rank': 4, 'alpha': 8, 'dropout': 0.05},     # 高alpha
        {'rank': 4, 'alpha': 16, 'dropout': 0.05},    # 很高alpha
       
        
        # === Dropout系列实验 (固定rank=4, alpha=1) ===
        {'rank': 4, 'alpha': 1, 'dropout': 0.0},      # 无dropout
        {'rank': 4, 'alpha': 1, 'dropout': 0.025},    # 极低dropout
        {'rank': 4, 'alpha': 1, 'dropout': 0.05},     # 基准dropout (重复，用作参考)
        {'rank': 4, 'alpha': 1, 'dropout': 0.1},      # 中dropout
        {'rank': 4, 'alpha': 1, 'dropout': 0.15},     # 高dropout
        {'rank': 4, 'alpha': 1, 'dropout': 0.2},      # 很高dropout
        {'rank': 4, 'alpha': 1, 'dropout': 0.3},      # 极高dropout
        
        
    ]
    
    print(f"定义了 {len(selected_configs)} 个超参数配置")
    for i, config in enumerate(selected_configs):
        print(f"{i+1}: r={config['rank']}, alpha={config['alpha']}, dropout={config['dropout']}")
    
    # 如果想要快速实验，可以使用核心配置子集
    # core_configs = selected_configs[0:15]  # 只使用前15个配置
    # print(f"\n注意: 如果时间有限，可以只运行前15个核心配置")
    
    # 进行超参数实验（IMDB数据集）
    print("=== 开始IMDB数据集超参数实验 ===")
    
    imdb_results = []
    
    for i, hyperparameter_config in enumerate(selected_configs):
        print(f"\n{'='*50}")
        print(f"实验 {i+1}/{len(selected_configs)}")
        
        try:
            experiment_result = train_lora_with_params(
                rank=hyperparameter_config['rank'],
                alpha=hyperparameter_config['alpha'], 
                dropout=hyperparameter_config['dropout'],
                dataset=imdb_train_data,
                dataset_name="imdb",
                model_name=model_name,
                epochs=2
            )
            imdb_results.append(experiment_result)
            
            print(f"✓ 完成 - 最终损失: {experiment_result['final_loss']:.4f}")
            
        except Exception as e:
            print(f"✗ 失败: {e}")
            continue

    print(f"\n=== IMDB超参数实验完成 ===")
    print(f"成功完成 {len(imdb_results)} 个实验")
    
    # 分析IMDB数据集超参数实验结果
    imdb_best_config = analyze_hyperparameter_results(imdb_results, "IMDB")
    
    # 得分点4：基于Alpaca数据集进行LoRA微调
    print("\n=== 得分点4：基于Alpaca数据集进行LoRA微调 ===")
    
    # 加载Alpaca数据集
    print("=== 加载Alpaca数据集 ===")
    alpaca_train_data = load_alpaca_dataset(tokenizer, num_samples=100)
    print(f"Alpaca数据集加载完成，样本数: {len(alpaca_train_data)}")
    
    # 查看Alpaca数据集样本
    print("\n=== Alpaca数据集样本 ===")
    sample_text = alpaca_train_data[0]
    print(f"输入长度: {len(sample_text['input_ids'])}")
    print(f"样本文本片段: {tokenizer.decode(sample_text['input_ids'][:200])}...")
    
    # 使用IMDB的最佳配置作为起点训练Alpaca数据集
    if imdb_best_config is not None:
        best_rank = imdb_best_config['rank']
        best_alpha = imdb_best_config['alpha']
        best_dropout = imdb_best_config['dropout']
        print(f"\n使用IMDB最佳配置: r={best_rank}, alpha={best_alpha}, dropout={best_dropout}")
    else:
        # 使用默认配置
        best_rank, best_alpha, best_dropout = 4, 1, 0.05
        print(f"\n使用默认配置: r={best_rank}, alpha={best_alpha}, dropout={best_dropout}")

    # 使用最佳配置训练Alpaca数据集
    print("\n=== 使用最佳配置训练Alpaca数据集 ===")
    alpaca_best_result = train_lora_with_params(
        rank=best_rank,
        alpha=best_alpha,
        dropout=best_dropout,
        dataset=alpaca_train_data,
        dataset_name="alpaca",
        model_name=model_name,
        epochs=2
    )
    
    print(f"Alpaca训练完成 - 最终损失: {alpaca_best_result['final_loss']:.4f}")
    
    # 得分点5：基于Alpaca数据集调试lora超参数，判断两个数据集对应的超参数是否有差异
    print("\n=== 得分点5：Alpaca数据集超参数调试和对比分析 ===")
    
    # 进行Alpaca数据集的完整超参数实验
    print("=== 开始Alpaca数据集超参数实验 ===")
    
    alpaca_results = []
    
    for i, hyperparameter_config in enumerate(selected_configs):
        print(f"\n{'='*50}")
        print(f"Alpaca实验 {i+1}/{len(selected_configs)}")
        
        try:
            experiment_result = train_lora_with_params(
                rank=hyperparameter_config['rank'],
                alpha=hyperparameter_config['alpha'], 
                dropout=hyperparameter_config['dropout'],
                dataset=alpaca_train_data,
                dataset_name="alpaca",
                model_name=model_name,
                epochs=2  # 使用相同的epoch数进行对比
            )
            alpaca_results.append(experiment_result)
            
            print(f"✓ 完成 - 最终损失: {experiment_result['final_loss']:.4f}")
            
        except Exception as e:
            print(f"✗ 失败: {e}")
            continue

    print(f"\n=== Alpaca超参数实验完成 ===")
    print(f"成功完成 {len(alpaca_results)} 个实验")
    
    # 分析Alpaca数据集超参数实验结果
    alpaca_best_config = analyze_hyperparameter_results(alpaca_results, "Alpaca")
    
    # 对比两个数据集的超参数结果
    print("\n" + "="*80)
    print("=== 两个数据集超参数对比分析 ===")
    print("="*80)
    
    if imdb_best_config is not None and alpaca_best_config is not None:
        print("\n【最佳超参数对比】")
        print(f"IMDB最佳配置:   r={imdb_best_config['rank']}, alpha={imdb_best_config['alpha']}, dropout={imdb_best_config['dropout']}, loss={imdb_best_config['final_loss']:.4f}")
        print(f"Alpaca最佳配置: r={alpaca_best_config['rank']}, alpha={alpaca_best_config['alpha']}, dropout={alpaca_best_config['dropout']}, loss={alpaca_best_config['final_loss']:.4f}")
        
        # 分析差异
        rank_diff = alpaca_best_config['rank'] - imdb_best_config['rank']
        alpha_diff = alpaca_best_config['alpha'] - imdb_best_config['alpha']
        dropout_diff = alpaca_best_config['dropout'] - imdb_best_config['dropout']
        
        print(f"\n【超参数差异分析】")
        print(f"Rank差异:    {rank_diff:+d} ({'Alpaca需要更高rank' if rank_diff > 0 else 'IMDB需要更高rank' if rank_diff < 0 else '相同'})")
        print(f"Alpha差异:   {alpha_diff:+.0f} ({'Alpaca需要更高alpha' if alpha_diff > 0 else 'IMDB需要更高alpha' if alpha_diff < 0 else '相同'})")
        print(f"Dropout差异: {dropout_diff:+.2f} ({'Alpaca需要更高dropout' if dropout_diff > 0 else 'IMDB需要更高dropout' if dropout_diff < 0 else '相同'})")
        
        # 绘制对比图表
        compare_datasets_hyperparameters(imdb_results, alpaca_results, imdb_best_config, alpaca_best_config)
        
        # 深入分析原因
        print(f"\n【差异原因分析】")
        analyze_hyperparameter_differences(imdb_best_config, alpaca_best_config, imdb_results, alpaca_results)
        
    else:
        print("\n无法进行对比分析 - 缺少有效的实验结果")
    
    # 对比两个数据集的训练结果
    print("\n=== 最终训练结果对比 ===")
    print(f"IMDB最佳配置损失: {imdb_best_config['final_loss']:.4f}" if imdb_best_config else "IMDB: 无有效结果")
    print(f"Alpaca最佳配置损失: {alpaca_best_config['final_loss']:.4f}" if alpaca_best_config else "Alpaca: 无有效结果")
    
    print("\n=== 实验完成 ===")
    
    # 测试Alpaca训练后的模型
    print("=== 测试Alpaca训练后的模型 ===")
    final_alpaca_model = PeftModel.from_pretrained(foundation_model, alpaca_best_config['model_path'], is_trainable=False)
    
    instruction_prompts = [
        "### Instruction:\nWrite a short story about AI.\n\n### Response:\n",
        "### Instruction:\nExplain photosynthesis.\n\n### Response:\n",
        "### Instruction:\nWrite a haiku about coding.\n\n### Response:\n"
    ]
    
    print("Alpaca模型指令跟随测试:")
    for i, prompt in enumerate(instruction_prompts, 1):
        print(f"\n{i}. 指令: {prompt.split('Response:')[0]}Response:")
        response = test_model_generation(final_alpaca_model, prompt, tokenizer)
        # 只显示生成的部分
        generated_part = response[len(prompt):] if len(response) > len(prompt) else response
        print(f"   生成: {generated_part}")
    
    print("得分点5：基于Alpaca数据集调试lora超参数，判断两个数据集对应的超参数是否有差异，并尝试分析原因")
    
    

if __name__ == "__main__":
    main()