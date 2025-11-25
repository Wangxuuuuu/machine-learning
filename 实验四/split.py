import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """加载数据文件"""
    raw_data = np.loadtxt(filename)
    X = raw_data[:, :256]
    y_one_hot = raw_data[:, 256:]
    y = np.argmax(y_one_hot, axis=1)
    return X, y

def stratified_split(X, y, test_size=0.3, random_state=42):
    """分层采样函数"""
    np.random.seed(random_state)
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    
    for class_label in range(10):
        class_indices = np.where(y == class_label)[0]
        np.random.shuffle(class_indices)
        n_test = int(len(class_indices) * test_size)
        
        test_indices = class_indices[:n_test]
        train_indices = class_indices[n_test:]
        
        X_train_list.append(X[train_indices])
        X_test_list.append(X[test_indices])
        y_train_list.append(y[train_indices])
        y_test_list.append(y[test_indices])
    
    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.hstack(y_train_list)
    y_test = np.hstack(y_test_list)
    
    # 打乱顺序
    train_shuffle = np.random.permutation(len(X_train))
    test_shuffle = np.random.permutation(len(X_test))
    
    return X_train[train_shuffle], X_test[test_shuffle], y_train[train_shuffle], y_test[test_shuffle]

# 主程序
if __name__ == "__main__":
    # 1. 加载数据
    X, y = load_data('semeion.data.txt')
    
    # 2. 分层采样
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    
    # 3. 验证结果
    print("数据划分完成！")
    print(f"训练集: {len(X_train)} 个样本")
    print(f"测试集: {len(X_test)} 个样本")
    
    # 4. 保存数据
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)