import numpy as np
import matplotlib.pyplot as plt

def detailed_validation():
    """详细验证数据划分的质量"""
    
    # 加载保存的数据
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    print("=== 数据划分详细验证 ===")
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    print(f"总样本数: {len(X_train) + len(X_test)}")
    print(f"训练集比例: {len(X_train)/(len(X_train)+len(X_test)):.3f}")
    print(f"测试集比例: {len(X_test)/(len(X_train)+len(X_test)):.3f}")
    
    print("\n=== 各类别分布验证 ===")
    print("数字\t训练集\t测试集\t训练比例\t测试比例")
    print("-" * 50)
    
    for digit in range(10):
        train_count = np.sum(y_train == digit)
        test_count = np.sum(y_test == digit)
        train_ratio = train_count / len(y_train)
        test_ratio = test_count / len(y_test)
        
        print(f"{digit}\t{train_count}\t{test_count}\t{train_ratio:.3f}\t\t{test_ratio:.3f}")
    
# 运行验证
detailed_validation()