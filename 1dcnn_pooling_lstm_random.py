# 文件名：final_all_models_random_search_optimized.py
import numpy as np
import os
import tensorflow as tf
import time
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dropout, Dense, 
    BatchNormalization, LSTM, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ================= 配置参数 =================
unique_labels = 50
PROCESSED_DATA_FOLDER = "./processed_data"
TIME_STEPS = 16
RANDOM_SEARCH_ITER = 1  # 每次运行处理10个参数组
RESULTS_FILE = "random_search_results.csv"

# ================= 超参数空间 =================
HYPERPARAM_GRID = {
    'filters': [32, 64, 96, 128, 192, 256,384,512,768,1024],
    'kernel_size': [1,2, 3, 5,8,10,15],
    'pool_size': [1, 2, 3, 5,8,10,15],
    'dropout_rate': [0,0.1,0.2, 0.3, 0.4, 0.5, 0.6,0.7],
    'dense_units': [32, 64, 96, 128, 192, 256,384,512,768,1024]
}

# ================= 核心功能函数 =================
def split_dataset(X, y):
    """数据集划分"""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def calculate_top3_accuracy(y_true, y_pred_probs):
    """计算Top3准确率"""
    top3 = tf.math.top_k(y_pred_probs, k=3).indices.numpy()
    return np.mean([y_true[i] in top3[i] for i in range(len(y_true))])

# ================= 模型构建 =================
def build_hyper_model(input_shape, params):
    """可配置超参数的1D-CNN+LSTM模型"""
    return Sequential([
        Input(shape=input_shape),
        Conv1D(params['filters'], params['kernel_size'], 
              activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(params['pool_size']),
        Dropout(params['dropout_rate']),
        LSTM(128, return_sequences=False),
        Dense(params['dense_units'], activation='relu'),
        Dense(unique_labels, activation='softmax')
    ])

# ================= 参数管理模块 =================
def load_existing_params():
    """加载已有参数记录"""
    if not os.path.exists(RESULTS_FILE):
        return []
    
    try:
        existing_df = pd.read_csv(RESULTS_FILE)
        required_columns = ['filters', 'kernel_size', 'pool_size', 'dropout_rate', 'dense_units']
        
        # 校验文件格式
        if not all(col in existing_df.columns for col in required_columns):
            print("检测到不完整的结果文件，将创建新文件")
            return []
            
        # 精确类型转换
        existing_params = []
        for _, row in existing_df.iterrows():
            param = {
                'filters': int(row['filters']),
                'kernel_size': int(row['kernel_size']),
                'pool_size': int(row['pool_size']),
                'dropout_rate': float(row['dropout_rate']),
                'dense_units': int(row['dense_units'])
            }
            existing_params.append(param)
            
        return existing_params
        
    except Exception as e:
        print(f"加载历史参数失败: {str(e)}，将创建新文件")
        return []

def is_param_existing(new_param, existing_params):
    """精确检查参数是否存在"""
    for exist_param in existing_params:
        if (exist_param['filters'] == new_param['filters'] and
            exist_param['kernel_size'] == new_param['kernel_size'] and
            exist_param['pool_size'] == new_param['pool_size'] and
            np.isclose(exist_param['dropout_rate'], new_param['dropout_rate']) and
            exist_param['dense_units'] == new_param['dense_units']):
            return True
    return False

def generate_model_builders(n_iter, existing_params):
    """智能参数生成器"""
    builders = []
    generated_params = []
    MAX_ATTEMPTS = 1000  # 防死循环保护
    
    for _ in range(n_iter):
        attempt = 0
        while True:
            # 生成候选参数
            candidate = {
                'filters': random.choice(HYPERPARAM_GRID['filters']),
                'kernel_size': random.choice(HYPERPARAM_GRID['kernel_size']),
                'pool_size': random.choice(HYPERPARAM_GRID['pool_size']),
                'dropout_rate': random.choice(HYPERPARAM_GRID['dropout_rate']),
                'dense_units': random.choice(HYPERPARAM_GRID['dense_units'])
            }
            
            # 检查参数唯一性
            if (not is_param_existing(candidate, existing_params) and 
                not is_param_existing(candidate, generated_params)):
                generated_params.append(candidate)
                break
                
            attempt += 1
            if attempt >= MAX_ATTEMPTS:
                raise RuntimeError("无法生成新参数，请扩大超参数范围或减少搜索次数")
        
        # 创建模型构建器
        def model_builder(input_shape, p=candidate):  # 使用闭包固定参数
            model = build_hyper_model(input_shape, p)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
            
        model_builder.__name__ = (
            f"build_f{candidate['filters']}_k{candidate['kernel_size']}"
            f"_p{candidate['pool_size']}_dr{candidate['dropout_rate']:.1f}"
            f"_du{candidate['dense_units']}"
        ).replace('.', '')
        model_builder.params = candidate
        
        builders.append(model_builder)
    
    return builders

# ================= 训练流程 =================
def preprocess_for_temporal(X_train, X_val, X_test):
    """时序数据预处理"""
    h, w, c = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    features_per_step = h * w * c // TIME_STEPS
    return (
        X_train.reshape(-1, TIME_STEPS, features_per_step),
        X_val.reshape(-1, TIME_STEPS, features_per_step),
        X_test.reshape(-1, TIME_STEPS, features_per_step)
    )

def train_once(model_builder, X_train, X_val, X_test, y_train, y_val, y_test, run_id):
    """单次训练评估流程"""
    result = {
        'train_time': None,
        'top1_acc': None,
        'top3_acc': None,
        'error': None
    }
    
    try:
        # 数据预处理
        X_train_proc, X_val_proc, X_test_proc = preprocess_for_temporal(X_train, X_val, X_test)
        
        # 模型构建
        model = model_builder(X_train_proc.shape[1:])
        
        # 数据编码
        y_train_one_hot = to_categorical(y_train, num_classes=unique_labels)
        y_val_one_hot = to_categorical(y_val, num_classes=unique_labels)
        
        # 类别权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i:w for i,w in enumerate(class_weights)}
        
        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)
        ]
        
        # 训练过程
        start_time = time.time()
        history = model.fit(
            X_train_proc, y_train_one_hot,
            epochs=150,
            validation_data=(X_val_proc, y_val_one_hot),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=0
        )
        
        # 保存训练日志
        log_dir = "random_search_logs"
        os.makedirs(log_dir, exist_ok=True)
        model_name = model_builder.__name__[6:]
        log_filename = os.path.join(log_dir, f"{model_name}_run{run_id}_log.csv")
        pd.DataFrame(history.history).to_csv(log_filename, index=False)
        
        # 评估模型
        test_pred = model.predict(X_test_proc, verbose=0)
        y_pred = np.argmax(test_pred, axis=1)
        
        result.update({
            'train_time': round(time.time()-start_time, 1),
            'top1_acc': round(accuracy_score(y_test, y_pred)*100, 2),
            'top3_acc': round(calculate_top3_accuracy(y_test, test_pred)*100, 2)
        })
        
    except Exception as e:
        result['error'] = str(e)
    
    finally:
        if 'model' in locals():
            del model
        tf.keras.backend.clear_session()
        import gc; gc.collect()
    
    return result

# ================= 主程序 =================
if __name__ == "__main__":
    # 固定随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # 加载数据
    print("加载预处理数据...")
    try:
        X = np.load(os.path.join(PROCESSED_DATA_FOLDER, "X.npy"))
        y = np.load(os.path.join(PROCESSED_DATA_FOLDER, "Y.npy"))
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit(1)
    
    # 数据集划分
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    
    # 参数管理
    existing_params = load_existing_params()
    print(f"已加载历史参数记录: {len(existing_params)} 条")
    
    # 生成模型构建器
    try:
        models_to_test = generate_model_builders(RANDOM_SEARCH_ITER, existing_params)
    except RuntimeError as e:
        print(f"\033[31m错误：{str(e)}\033[0m")
        exit(1)
        
    # 训练流程
    print(f"\n开始随机搜索，共{RANDOM_SEARCH_ITER}次试验，每个参数运行三次...")
    results = []
    for i, builder in enumerate(models_to_test, 1):
        print(f"\n试验进度: {i}/{RANDOM_SEARCH_ITER} | 当前模型: {builder.__name__[6:]}")
        
        # 三次训练
        result_run1 = train_once(builder, X_train, X_val, X_test, y_train, y_val, y_test, run_id=1)
        result_run2 = train_once(builder, X_train, X_val, X_test, y_train, y_val, y_test, run_id=2)
        result_run3 = train_once(builder, X_train, X_val, X_test, y_train, y_val, y_test, run_id=3)
        
        # 合并结果
        combined_result = {
            'model_name': builder.__name__[6:],
            'params': builder.params,
            'run1_top1': result_run1['top1_acc'] if result_run1['error'] is None else None,
            'run1_top3': result_run1['top3_acc'] if result_run1['error'] is None else None,
            'run1_time': result_run1['train_time'] if result_run1['error'] is None else None,
            'run2_top1': result_run2['top1_acc'] if result_run2['error'] is None else None,
            'run2_top3': result_run2['top3_acc'] if result_run2['error'] is None else None,
            'run2_time': result_run2['train_time'] if result_run2['error'] is None else None,
            'run3_top1': result_run3['top1_acc'] if result_run3['error'] is None else None,
            'run3_top3': result_run3['top3_acc'] if result_run3['error'] is None else None,
            'run3_time': result_run3['train_time'] if result_run3['error'] is None else None,
            'error_run1': result_run1['error'],
            'error_run2': result_run2['error'],
            'error_run3': result_run3['error'],
        }
        
        # 计算统计量
        top1_values = [v for v in [combined_result['run1_top1'], combined_result['run2_top1'], combined_result['run3_top1']] if v is not None]
        top3_values = [v for v in [combined_result['run1_top3'], combined_result['run2_top3'], combined_result['run3_top3']] if v is not None]
        time_values = [v for v in [combined_result['run1_time'], combined_result['run2_time'], combined_result['run3_time']] if v is not None]
        
        combined_result['top1_avg'] = np.mean(top1_values) if top1_values else None
        combined_result['top1_std'] = np.std(top1_values) if len(top1_values) > 1 else 0.0
        combined_result['top3_avg'] = np.mean(top3_values) if top3_values else None
        combined_result['top3_std'] = np.std(top3_values) if len(top3_values) > 1 else 0.0
        combined_result['time_avg'] = np.mean(time_values) if time_values else None
        combined_result['time_std'] = np.std(time_values) if len(time_values) > 1 else 0.0
        
        results.append(combined_result)
        
        # 打印当前结果
        if combined_result['error_run1'] or combined_result['error_run2'] or combined_result['error_run3']:
            err_msg = f"训练出现错误: Run1-{combined_result['error_run1']}, Run2-{combined_result['error_run2']}, Run3-{combined_result['error_run3']}"
            print(f"\033[31m{err_msg}\033[0m")
        else:
            print(f"Run1: Top1={combined_result['run1_top1']}%, Top3={combined_result['run1_top3']}%, Time={combined_result['run1_time']}s")
            print(f"Run2: Top1={combined_result['run2_top1']}%, Top3={combined_result['run2_top3']}%, Time={combined_result['run2_time']}s")
            print(f"Run3: Top1={combined_result['run3_top1']}%, Top3={combined_result['run3_top3']}%, Time={combined_result['run3_time']}s")
            print(f"平均: Top1={combined_result['top1_avg']:.2f}±{combined_result['top1_std']:.2f}%, "
                  f"Top3={combined_result['top3_avg']:.2f}±{combined_result['top3_std']:.2f}%, "
                  f"Time={combined_result['time_avg']:.1f}±{combined_result['time_std']:.1f}s")
    
    # 保存结果
    if results:
        success_entries = []
        for res in results:
            entry = {
                **res['params'],
                'Model': res['model_name'],
                'Run1_Top1': res['run1_top1'],
                'Run2_Top1': res['run2_top1'],
                'Run3_Top1': res['run3_top1'],
                'Top1_Avg': res['top1_avg'],
                'Top1_Std': res['top1_std'],
                'Run1_Top3': res['run1_top3'],
                'Run2_Top3': res['run2_top3'],
                'Run3_Top3': res['run3_top3'],
                'Top3_Avg': res['top3_avg'],
                'Top3_Std': res['top3_std'],
                'Run1_Time': res['run1_time'],
                'Run2_Time': res['run2_time'],
                'Run3_Time': res['run3_time'],
                'Time_Avg': res['time_avg'],
                'Time_Std': res['time_std'],
                'Error_Run1': res['error_run1'],
                'Error_Run2': res['error_run2'],
                'Error_Run3': res['error_run3']
            }
            success_entries.append(entry)
        
        success_df = pd.DataFrame(success_entries)
        
        # 合并历史记录
        if os.path.exists(RESULTS_FILE):
            history_df = pd.read_csv(RESULTS_FILE)
            success_df = pd.concat([history_df, success_df], ignore_index=True)
            
        # 去重保存（基于参数组合）
        success_df.drop_duplicates(
            subset=['filters', 'kernel_size', 'pool_size', 'dropout_rate', 'dense_units'],
            keep='first'
        ).to_csv(RESULTS_FILE, index=False)
        print(f"\n结果已保存至 {RESULTS_FILE}")
        
        # 显示最佳模型（按Top1平均）
        if not success_df.empty:
            success_df_clean = success_df.dropna(subset=['Top1_Avg'])
            if not success_df_clean.empty:
                best_idx = success_df_clean['Top1_Avg'].idxmax()
                best_model = success_df_clean.loc[best_idx]
                print("\n\033[36m=== 最佳模型 ===")
                print(f"模型名称: {best_model['Model']}")
                print(f"参数配置: filters={best_model['filters']}, kernel={best_model['kernel_size']}, "
                      f"pool={best_model['pool_size']}, dropout={best_model['dropout_rate']}, "
                      f"dense={best_model['dense_units']}")
                print(f"Top1准确率: {best_model['Run1_Top1']}% (Run1), {best_model['Run2_Top1']}% (Run2), {best_model['Run3_Top1']}% (Run3)")
                print(f"Top3准确率: {best_model['Run1_Top3']}% (Run1), {best_model['Run2_Top3']}% (Run2), {best_model['Run3_Top3']}% (Run3)")
                print(f"平均性能: Top1={best_model['Top1_Avg']:.2f}±{best_model['Top1_Std']:.2f}%, "
                      f"Top3={best_model['Top3_Avg']:.2f}±{best_model['Top3_Std']:.2f}%, "
                      f"时间={best_model['Time_Avg']:.1f}±{best_model['Time_Std']:.1f}s\033[0m")