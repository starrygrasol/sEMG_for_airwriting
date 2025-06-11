# 文件名：final_all_models.py
import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D,
    Flatten, Dropout, Dense, BatchNormalization,
    LSTM, Reshape, Bidirectional,GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
# 配置参数
unique_labels = 50
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
PROCESSED_DATA_FOLDER = "./processed_data"
TIME_STEPS = 16  # 必须与数据通道维度一致

# ================= 核心功能函数 =================
def split_dataset(X, y):
    """数据集划分"""
    print("\n" + "="*20 + " 开始数据集划分 " + "="*20)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=42)
    val_adjusted_ratio = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_adjusted_ratio, stratify=y_train_val, random_state=42)
    print(f"数据集划分完成 - 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    print("="*20 + " 数据集划分完成 " + "="*20 + "\n")
    return X_train, X_val, X_test, y_train, y_val, y_test

def calculate_top3_accuracy(y_true, y_pred_probs):
    """计算Top3准确率"""
    top3 = tf.math.top_k(y_pred_probs, k=3).indices.numpy()
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top3[i]:
            correct += 1
    return correct / len(y_true)

# ================= 模型定义 =================
def build_deeper_model(input_shape):
    """深度CNN模型"""
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(64, (3,3), activation='relu', padding='same', 
              kernel_regularizer=l2(1e-4))(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3,3), activation='relu', padding='same',
              kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(256, (3,3), activation='relu', padding='same',
              kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2,2)(x)
    
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    output = Dense(unique_labels, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_2dcnn(input_shape):
    """修复2DCNN模型"""
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4), input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Conv2D(128, (3,3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Conv2D(256, (3,3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        GlobalAveragePooling2D(),  # 修复2：已正确导入
        
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(unique_labels, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(input_shape):
    """修正后的LSTM模型"""
    model = Sequential([
        Input(shape=input_shape),  # 输入形状应为(16, 4096)
        LSTM(128, return_sequences=False),  # 直接处理时序数据
        Dense(64, activation='relu'),
        Dense(unique_labels, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_1dcnn_lstm(input_shape):
    """修复后的1D-CNN+LSTM模型"""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(unique_labels, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_1dcnn_pool_lstm(input_shape):
    """修复后的1D-CNN+Pooling+LSTM模型"""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(unique_labels, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_1dcnn_pool_lstm_improve(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.5),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(unique_labels, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ================= 统一训练流程 =================
def preprocess_for_temporal(X_train, X_val, X_test):
    """修复数据预处理"""
    # 转换为 (samples, timesteps, features)
    X_train_processed = X_train.transpose(0, 3, 1, 2).reshape(X_train.shape[0], TIME_STEPS, -1)
    X_val_processed = X_val.transpose(0, 3, 1, 2).reshape(X_val.shape[0], TIME_STEPS, -1)
    X_test_processed = X_test.transpose(0, 3, 1, 2).reshape(X_test.shape[0], TIME_STEPS, -1)
    return X_train_processed, X_val_processed, X_test_processed

def train_and_evaluate(model_builder, X_train, X_val, X_test, y_train, y_val, y_test):
    result = {
        'model_name': model_builder.__name__[6:],
        'train_time': None,
        'top1_acc': None,
        'top3_acc': None,
        'history': None,
        'error': None
    }
    
    try:
        # 调试：打印输入数据形状
        print(f"[DEBUG] X_train原始形状: {X_train.shape}")
        print(f"[DEBUG] y_train形状: {y_train.shape}")

        # 判断模型类型
        model_name = model_builder.__name__[6:].lower()
        is_temporal = 'lstm' in model_name or '1dcnn' in model_name
        
        # 时序模型特殊处理
        if is_temporal:
            try:
                # 显式获取维度并验证
                h, w, c = X_train.shape[1], X_train.shape[2], X_train.shape[3]
                print(f"[DEBUG] 时序模型维度检查 h={h}, w={w}, c={c}")
                
                total_features = h * w * c
                print(f"[DEBUG] 总特征数计算: {h}*{w}*{c}={total_features}")
                
                assert total_features % TIME_STEPS == 0, f"总特征数{total_features}无法被{TIME_STEPS}整除"
                features_per_step = total_features // TIME_STEPS
                input_shape = (TIME_STEPS, features_per_step)
                print(f"[DEBUG] 输入形状设置: {input_shape}")

                # 数据预处理
                X_train_processed = X_train.transpose(0, 3, 1, 2).reshape(-1, TIME_STEPS, features_per_step)
                X_val_processed = X_val.transpose(0, 3, 1, 2).reshape(-1, TIME_STEPS, features_per_step)
                X_test_processed = X_test.transpose(0, 3, 1, 2).reshape(-1, TIME_STEPS, features_per_step)
                print(f"[DEBUG] 预处理后形状 - 训练集: {X_train_processed.shape}, 验证集: {X_val_processed.shape}")
            except Exception as e:
                raise ValueError(f"时序数据预处理失败: {str(e)}") from e
        else:
            input_shape = X_train.shape[1:]
            X_train_processed, X_val_processed, X_test_processed = X_train, X_val, X_test

        # 模型构建
        try:
            model = model_builder(input_shape)
            model.summary()
        except Exception as e:
            raise ValueError(f"模型构建失败: {str(e)}") from e

 
        # 数据增强与数据准备
        if not is_temporal:
            try:
                datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    zoom_range=0.25,
                    horizontal_flip=True
                )
                train_generator = datagen.flow(X_train_processed, to_categorical(y_train, num_classes=unique_labels), batch_size=32)
                x_train = train_generator
                y_train_one_hot = None  # 生成器已包含标签
            except Exception as e:
                raise ValueError(f"数据增强失败: {str(e)}") from e
        else:
            # 时序模型直接使用处理后的数据和标签
            x_train = X_train_processed
            y_train_one_hot = to_categorical(y_train, num_classes=unique_labels)

        # 准备验证数据
        x_val = X_val_processed
        y_val_one_hot = to_categorical(y_val, num_classes=unique_labels)

        # 类别权重计算（原有代码不变）
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i:w for i,w in enumerate(class_weights)}

        # 回调函数（原有代码不变）
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)
        ]

        # 训练过程
        try:
            start_time = time.time()
            history = model.fit(
                x_train,
                y=y_train_one_hot,  # 时序模型传递标签，非时序模型为None
                epochs=150,
                validation_data=(x_val, y_val_one_hot),
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )
            result['history'] = history.history
            print("[DEBUG] 模型训练完成")
        except Exception as e:
            raise ValueError(f"训练过程失败: {str(e)}") from e

        # 评估过程
        try:
            test_pred = model.predict(X_test_processed)
            y_pred = np.argmax(test_pred, axis=1)  # 新增预测结果
            
            result.update({
                'train_time': round(time.time()-start_time, 1),
                'top1_acc': round(accuracy_score(y_test, y_pred)*100, 2),
                'top3_acc': round(calculate_top3_accuracy(y_test, test_pred)*100, 2),
                'y_test': y_test,  # 保存真实标签
                'y_pred': y_pred   # 保存预测标签
            })
        except Exception as e:
            raise ValueError(f"评估过程失败: {str(e)}") from e

    except Exception as e:
        result['error'] = str(e)
        print(f"\033[31m[ERROR] {result['model_name']} 失败阶段: {str(e)}\033[0m")
        import traceback
        traceback.print_exc()  # 打印完整堆栈跟踪
    
    return result
# ================= 主程序 =================
if __name__ == "__main__":
    # 加载数据
    print("\n" + "="*20 + " 加载预处理数据 " + "="*20)
    X = np.load(os.path.join(PROCESSED_DATA_FOLDER, "X.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_FOLDER, "Y.npy"))
    print(f"输入形状: {X.shape}, 标签形状: {y.shape}")

    # 数据集划分
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # 模型列表
    models_to_test = [
        build_deeper_model,
        build_2dcnn,
        build_lstm_model,
        build_1dcnn_lstm,
        build_1dcnn_pool_lstm,
        #build_1dcnn_pool_lstm_improve
    ]

    # 训练与评估
    results = []
    for builder in models_to_test:
        print(f"\n\033[34m=== 正在训练 {builder.__name__[6:]} ===\033[0m")
        res = train_and_evaluate(builder, X_train, X_val, X_test, y_train, y_val, y_test)
        results.append(res)

    # 打印结果
    print("\n\033[32m=== 最终性能对比 ===\033[0m")
    print(f"{'模型名称':<20} | {'训练时间(s)':>10} | {'Top1准确率(%)':>12} | {'Top3准确率(%)':>12}")
    print("-"*60)
    for res in results:
        if res['error'] is None:
            print(f"{res['model_name']:<20} | {res['train_time']:>10} | {res['top1_acc']:>12.2f} | {res['top3_acc']:>12.2f}")
        else:
            print(f"\033[31m{res['model_name']:<20} | 失败: {res['error'][:50]}...\033[0m")

    # 详细分析
    for res in results:
        if res['model_name'] == '1dcnn_pool_lstm' and res['error'] is None:
            y_test = res['y_test']
            y_pred = res['y_pred']
            class_names = [str(i) for i in range(unique_labels)]

            # 生成混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            
            # 可视化混淆矩阵
            plt.figure(figsize=(25, 20))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 查找前20个错误对
            error_matrix = cm.copy()
            np.fill_diagonal(error_matrix, 0)
            flat_indices = np.argsort(error_matrix.flatten())[::-1][:50]  # 取前50确保足够数量
            
            print("\nTop 20 Error Pairs:")
            error_count = 0
            for idx in flat_indices:
                true_cls, pred_cls = np.unravel_index(idx, error_matrix.shape)
                if true_cls != pred_cls and error_matrix[true_cls, pred_cls] > 0:
                    print(f"True: {true_cls:2d} → Pred: {pred_cls:2d} | 错误次数: {error_matrix[true_cls, pred_cls]:3d}")
                    error_count += 1
                    if error_count >= 20:
                        break

            # 生成分类报告
            report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
            print("\nDetailed Classification Report:")
            print(report)

            # 保存指标
            report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
            report_df.to_csv('classification_report.csv', float_format='%.4f')
            
            # 保存错误矩阵
            error_df = pd.DataFrame(error_matrix, index=class_names, columns=class_names)
            error_df.to_csv('error_matrix.csv')
    # 可视化训练曲线
    plt.figure(figsize=(15,6))
    plt.subplot(121)
    for res in results:
        if res['history']:
            plt.plot(res['history']['accuracy'], label=res['model_name'])
    plt.title('training accuracy')
    plt.legend()

    plt.subplot(122)
    for res in results:
        if res['history']:
            plt.plot(res['history']['loss'], label=res['model_name'])
    plt.title('training loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.show()