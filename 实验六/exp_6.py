import pandas as pd
import numpy as np
import math
from collections import Counter

class ID3DecisionTree:
    def __init__(self):
        self.tree = None
        self.feature_names = None
    
    def load_data(self, train_file, test_file):
        """加载训练集和测试集数据"""
        # 读取训练数据
        train_data = pd.read_csv(train_file, encoding='gbk')
        test_data = pd.read_csv(test_file, encoding='gbk')
        
        # 移除编号列，保留特征和标签
        self.X_train = train_data.iloc[:, 1:-1]  # 特征：从第2列到倒数第2列
        self.y_train = train_data.iloc[:, -1]    # 标签：最后一列
        self.X_test = test_data.iloc[:, 1:-1]
        self.y_test = test_data.iloc[:, -1]
        
        # 保存特征名称
        self.feature_names = list(self.X_train.columns)
        
        print("训练集形状:", self.X_train.shape)
        print("测试集形状:", self.X_test.shape)
        print("特征名称:", self.feature_names)
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def calculate_entropy(self, y):
        """
        计算信息熵
        参数: y - 标签序列
        返回: 熵值
        """
        # 统计每个类别的数量
        unique_labels, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        entropy = 0.0
        
        for count in counts:
            # 计算每个类别的概率
            probability = count / total_samples
            # 使用log2计算熵，避免log(0)的情况
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def calculate_information_gain(self, X, y, feature):
        """
        计算某个特征的信息增益
        参数: 
            X - 特征数据
            y - 标签数据  
            feature - 特征名称
        返回: 信息增益值
        """
        # 计算原始数据集的熵
        total_entropy = self.calculate_entropy(y)
        
        # 获取该特征的所有取值
        feature_values = X[feature].unique()
        
        # 计算加权平均熵
        weighted_entropy = 0.0
        total_samples = len(y)
        
        for value in feature_values:
            # 获取该特征值对应的子集
            subset_mask = X[feature] == value
            y_subset = y[subset_mask]
            
            # 计算子集的熵
            subset_entropy = self.calculate_entropy(y_subset)
            # 计算子集的权重
            subset_weight = len(y_subset) / total_samples
            
            weighted_entropy += subset_weight * subset_entropy
        
        # 信息增益 = 原始熵 - 加权平均熵
        information_gain = total_entropy - weighted_entropy
        return information_gain  

    def choose_best_feature(self, X, y, features):
        """
        选择信息增益最大的特征作为划分特征
        参数:
            X - 特征数据
            y - 标签数据
            features - 可用的特征列表
        返回: 最佳特征名称
        """
        best_gain = -1
        best_feature = None
        
        for feature in features:
            gain = self.calculate_information_gain(X, y, feature)
            print(f"特征 '{feature}' 的信息增益: {gain:.4f}")
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        print(f"选择最佳划分特征: {best_feature}, 信息增益: {best_gain:.4f}")
        return best_feature   

    def build_tree(self, X, y, features, depth=0):
        """
        递归构建决策树
        参数:
            X - 特征数据
            y - 标签数据
            features - 可用的特征列表
            depth - 当前深度（用于调试）
        返回: 决策树节点
        """
        # 终止条件1: 所有样本属于同一类别
        if len(np.unique(y)) == 1:
            print(f"深度{depth}: 所有样本属于同一类别 '{y.iloc[0]}'")
            return y.iloc[0]
        
        # 终止条件2: 没有特征可用
        if len(features) == 0:
            majority_class = y.mode()[0]  # 返回出现次数最多的类别
            print(f"深度{depth}: 无特征可用，返回多数类 '{majority_class}'")
            return majority_class
        
        # 选择最佳划分特征
        best_feature = self.choose_best_feature(X, y, features)
        
        # 创建树节点
        tree = {best_feature: {}}
        
        # 从特征列表中移除已选特征
        remaining_features = [f for f in features if f != best_feature]
        
        # 按最佳特征的每个取值划分子集
        for value in X[best_feature].unique():
            # 创建子集
            subset_mask = X[best_feature] == value
            X_subset = X[subset_mask]
            y_subset = y[subset_mask]
            
            # 如果子集为空，返回父节点的多数类
            if len(y_subset) == 0:
                majority_class = y.mode()[0]
                tree[best_feature][value] = majority_class
                print(f"深度{depth}: 特征'{best_feature}'取值'{value}'的子集为空，返回多数类 '{majority_class}'")
            else:
                # 递归构建子树
                print(f"深度{depth}: 在特征'{best_feature}'取值'{value}'上递归构建子树")
                tree[best_feature][value] = self.build_tree(
                    X_subset, y_subset, remaining_features, depth + 1
                )
        
        return tree         

    def fit(self, X_train, y_train):
        """
        训练ID3决策树模型
        参数:
            X_train - 训练特征
            y_train - 训练标签
        """
        print("开始构建ID3决策树...")
        self.feature_names = list(X_train.columns)
        self.tree = self.build_tree(X_train, y_train, self.feature_names)
        print("决策树构建完成!")
        return self

    def predict_sample(self, sample, tree):
        """
        对单个样本进行预测
        参数:
            sample - 单个样本数据
            tree - 决策树
        返回: 预测类别
        """
        # 如果当前节点是叶节点（字符串类型）
        if not isinstance(tree, dict):
            return tree
        
        # 获取当前节点的特征
        feature = list(tree.keys())[0]
        feature_value = sample[feature]
        
        # 如果特征值在树的分支中
        if feature_value in tree[feature]:
            subtree = tree[feature][feature_value]
            return self.predict_sample(sample, subtree)
        else:
            # 如果遇到未知的特征值，返回None（在实际应用中可能需要处理）
            return None    

    def predict(self, X_test):
        """
        对测试集进行预测
        参数:
            X_test - 测试特征
        返回: 预测结果列表
        """
        if self.tree is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        predictions = []
        for i in range(len(X_test)):
            sample = X_test.iloc[i]
            prediction = self.predict_sample(sample, self.tree)
            predictions.append(prediction)
        
        return predictions

    def calculate_accuracy(self, y_true, y_pred):
        """
        计算分类准确率
        参数:
            y_true - 真实标签
            y_pred - 预测标签
        返回: 准确率
        """
        correct = 0
        total = len(y_true)
        
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                correct += 1
        
        accuracy = correct / total
        return accuracy    
    
def main_id3():
    # 创建ID3决策树实例
    id3_tree = ID3DecisionTree()
    
    # 加载数据
    X_train, y_train, X_test, y_test = id3_tree.load_data(
        "watermelon-train1.csv", 
        "watermelon-test1.csv"
    )
    
    # 训练模型
    id3_tree.fit(X_train, y_train)
    
    # 进行预测
    y_pred = id3_tree.predict(X_test)
    
    # 计算准确率
    accuracy = id3_tree.calculate_accuracy(y_test, y_pred)
    
    print(f"\n在测试集上的分类准确率: {accuracy:.2f}")
    
    # 输出预测结果对比
    print("\n预测结果对比:")
    print("真实标签:", list(y_test))
    print("预测标签:", y_pred)    


class C45DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.tree = None
        self.feature_names = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.continuous_features = []
        self.discrete_features = []
    
    def load_data(self, train_file, test_file):
        """加载训练集和测试集数据，识别连续和离散属性"""
        train_data = pd.read_csv(train_file, encoding='gbk')
        test_data = pd.read_csv(test_file, encoding='gbk')
        
        self.X_train = train_data.iloc[:, 1:-1]
        self.y_train = train_data.iloc[:, -1]
        self.X_test = test_data.iloc[:, 1:-1]
        self.y_test = test_data.iloc[:, -1]
        
        self.feature_names = list(self.X_train.columns)
        
        # 自动识别连续和离散属性
        self._identify_feature_types()
        
        print("训练集形状:", self.X_train.shape)
        print("测试集形状:", self.X_test.shape)
        print("特征名称:", self.feature_names)
        print("连续特征:", self.continuous_features)
        print("离散特征:", self.discrete_features)
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def _identify_feature_types(self):
        """识别连续和离散特征"""
        for feature in self.feature_names:
            # 如果特征值数量多且为数值型，则认为是连续特征
            unique_values = self.X_train[feature].unique()
            if (len(unique_values) > 10 and 
                pd.api.types.is_numeric_dtype(self.X_train[feature])):
                self.continuous_features.append(feature)
            else:
                self.discrete_features.append(feature)
    
    def calculate_entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y) if y.dtype == np.int64 else np.array(list(Counter(y).values()))
        probabilities = counts / len(y)
        entropy = -np.sum([p * math.log2(p) for p in probabilities if p > 0])
        return entropy
    
    def calculate_information_gain_ratio(self, X, y, feature):
        """
        计算信息增益率
        对于离散特征：使用标准的信息增益率
        对于连续特征：先离散化，然后计算信息增益率
        """
        if feature in self.discrete_features:
            return self._discrete_feature_gain_ratio(X, y, feature)
        else:
            return self._continuous_feature_gain_ratio(X, y, feature)
    
    def _discrete_feature_gain_ratio(self, X, y, feature):
        """计算离散特征的信息增益率"""
        total_entropy = self.calculate_entropy(y)
        
        feature_values = X[feature].unique()
        weighted_entropy = 0.0
        split_info = 0.0
        total_samples = len(y)
        
        for value in feature_values:
            subset_mask = X[feature] == value
            y_subset = y[subset_mask]
            subset_size = len(y_subset)
            
            if subset_size > 0:
                subset_entropy = self.calculate_entropy(y_subset)
                subset_weight = subset_size / total_samples
                weighted_entropy += subset_weight * subset_entropy
                
                # 计算分裂信息
                probability = subset_size / total_samples
                split_info -= probability * math.log2(probability)
        
        information_gain = total_entropy - weighted_entropy
        
        # 避免除零错误
        if split_info == 0:
            return 0
        
        gain_ratio = information_gain / split_info
        return gain_ratio
    
    def _continuous_feature_gain_ratio(self, X, y, feature):
        """计算连续特征的信息增益率（通过最佳划分点）"""
        # 对特征值排序
        sorted_indices = X[feature].sort_values().index
        unique_values = sorted(X[feature].unique())
        
        if len(unique_values) <= 1:
            return 0
        
        best_gain_ratio = 0
        best_split_point = None
        
        # 尝试所有可能的划分点（相邻值的中间点）
        for i in range(len(unique_values) - 1):
            split_point = (unique_values[i] + unique_values[i + 1]) / 2
            
            # 根据划分点创建虚拟的离散特征
            left_mask = X[feature] <= split_point
            right_mask = X[feature] > split_point
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            # 计算信息增益
            total_entropy = self.calculate_entropy(y)
            left_weight = len(y_left) / len(y)
            right_weight = len(y_right) / len(y)
            weighted_entropy = (left_weight * self.calculate_entropy(y_left) + 
                              right_weight * self.calculate_entropy(y_right))
            information_gain = total_entropy - weighted_entropy
            
            # 计算分裂信息（二分划分）
            split_info = - (left_weight * math.log2(left_weight) + 
                          right_weight * math.log2(right_weight))
            
            gain_ratio = information_gain / split_info if split_info > 0 else 0
            
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_split_point = split_point
        
        return best_gain_ratio, best_split_point
    
    def choose_best_feature(self, X, y, features):
        """选择信息增益率最大的特征"""
        best_gain_ratio = -1
        best_feature = None
        best_split_point = None
        
        for feature in features:
            if feature in self.discrete_features:
                gain_ratio = self.calculate_information_gain_ratio(X, y, feature)
                print(f"离散特征 '{feature}' 的信息增益率: {gain_ratio:.4f}")
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    best_split_point = None
            else:
                gain_ratio, split_point = self.calculate_information_gain_ratio(X, y, feature)
                print(f"连续特征 '{feature}' 的信息增益率: {gain_ratio:.4f}, 最佳划分点: {split_point}")
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    best_split_point = split_point
        
        print(f"选择最佳划分特征: {best_feature}, 信息增益率: {best_gain_ratio:.4f}")
        if best_split_point is not None:
            print(f"最佳划分点: {best_split_point}")
        
        return best_feature, best_split_point
    
    def build_tree(self, X, y, features, depth=0):
        """递归构建C4.5决策树"""
        # 终止条件检查
        if self._should_stop(y, features, depth):
            return self._get_leaf_value(y)
        
        # 选择最佳特征
        best_feature, split_point = self.choose_best_feature(X, y, features)
        
        if best_feature is None:
            return self._get_leaf_value(y)
        
        # 创建树节点
        tree = {'feature': best_feature, 'split_point': split_point, 'children': {}}
        
        # 从特征列表中移除已选特征（对于离散特征）
        remaining_features = [f for f in features if f != best_feature]
        
        # 根据特征类型进行划分
        if best_feature in self.discrete_features:
            # 离散特征：按每个取值划分
            for value in X[best_feature].unique():
                subset_mask = X[best_feature] == value
                X_subset = X[subset_mask]
                y_subset = y[subset_mask]
                
                if len(y_subset) == 0:
                    tree['children'][value] = self._get_leaf_value(y)
                else:
                    tree['children'][value] = self.build_tree(
                        X_subset, y_subset, remaining_features, depth + 1
                    )
        else:
            # 连续特征：按划分点二分
            left_mask = X[best_feature] <= split_point
            right_mask = X[best_feature] > split_point
            
            # 左子树（<= 划分点）
            X_left = X[left_mask]
            y_left = y[left_mask]
            if len(y_left) == 0:
                tree['children']['left'] = self._get_leaf_value(y)
            else:
                tree['children']['left'] = self.build_tree(
                    X_left, y_left, remaining_features, depth + 1
                )
            
            # 右子树（> 划分点）
            X_right = X[right_mask]
            y_right = y[right_mask]
            if len(y_right) == 0:
                tree['children']['right'] = self._get_leaf_value(y)
            else:
                tree['children']['right'] = self.build_tree(
                    X_right, y_right, remaining_features, depth + 1
                )
        
        return tree
    
    def _should_stop(self, y, features, depth):
        """检查是否应该停止分裂"""
        # 所有样本属于同一类别
        if len(np.unique(y)) == 1:
            return True
        
        # 没有特征可用
        if len(features) == 0:
            return True
        
        # 样本数小于最小分裂样本数
        if len(y) < self.min_samples_split:
            return True
        
        # 达到最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        
        return False
    
    def _get_leaf_value(self, y):
        """获取叶节点的值（多数类）"""
        return y.mode()[0] if len(y) > 0 else None
    
    def fit(self, X_train, y_train):
        """训练C4.5决策树模型"""
        print("开始构建C4.5决策树...")
        self.feature_names = list(X_train.columns)
        self._identify_feature_types()
        self.tree = self.build_tree(X_train, y_train, self.feature_names)
        print("C4.5决策树构建完成!")
        return self
    
    def predict_sample(self, sample, tree):
        """对单个样本进行预测"""
        if not isinstance(tree, dict):
            return tree
        
        feature = tree['feature']
        split_point = tree['split_point']
        
        if feature in self.discrete_features:
            # 离散特征
            feature_value = sample[feature]
            if feature_value in tree['children']:
                return self.predict_sample(sample, tree['children'][feature_value])
            else:
                # 处理未知特征值
                return None
        else:
            # 连续特征
            feature_value = sample[feature]
            if feature_value <= split_point:
                return self.predict_sample(sample, tree['children']['left'])
            else:
                return self.predict_sample(sample, tree['children']['right'])
    
    def predict(self, X_test):
        """对测试集进行预测"""
        if self.tree is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        predictions = []
        for i in range(len(X_test)):
            sample = X_test.iloc[i]
            prediction = self.predict_sample(sample, self.tree)
            predictions.append(prediction)
        
        return predictions
    
    def calculate_accuracy(self, y_true, y_pred):
        """计算分类准确率"""
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

def main_c45():
    """C4.5决策树主函数"""
    # 创建C4.5决策树实例
    c45_tree = C45DecisionTree(min_samples_split=2, max_depth=5)
    
    # 加载包含连续属性的数据
    X_train, y_train, X_test, y_test = c45_tree.load_data(
        "watermelon-train2.csv", 
        "watermelon-test2.csv"
    )
    
    # 训练模型
    c45_tree.fit(X_train, y_train)
    
    # 进行预测
    y_pred = c45_tree.predict(X_test)
    
    # 计算准确率
    accuracy = c45_tree.calculate_accuracy(y_test, y_pred)
    
    print(f"\n在测试集watermelon-test2上的分类准确率: {accuracy:.2f}")
    
    # 输出预测结果对比
    print("\n预测结果对比:")
    print("真实标签:", list(y_test))
    print("预测标签:", y_pred)






import pandas as pd
import numpy as np
import math
from collections import Counter

class CARTDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.tree = None
        self.feature_names = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.continuous_features = []
        self.discrete_features = []
    
    def load_data(self, train_file, test_file):
        """加载训练集和测试集数据，识别连续和离散属性"""
        train_data = pd.read_csv(train_file, encoding='gbk')
        test_data = pd.read_csv(test_file, encoding='gbk')
        
        self.X_train = train_data.iloc[:, 1:-1]
        self.y_train = train_data.iloc[:, -1]
        self.X_test = test_data.iloc[:, 1:-1]
        self.y_test = test_data.iloc[:, -1]
        
        self.feature_names = list(self.X_train.columns)
        
        # 自动识别连续和离散属性
        self._identify_feature_types()
        
        print("训练集形状:", self.X_train.shape)
        print("测试集形状:", self.X_test.shape)
        print("特征名称:", self.feature_names)
        print("连续特征:", self.continuous_features)
        print("离散特征:", self.discrete_features)
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def _identify_feature_types(self):
        """识别连续和离散特征"""
        for feature in self.feature_names:
            # 如果特征值数量多且为数值型，则认为是连续特征
            unique_values = self.X_train[feature].unique()
            if (len(unique_values) > 10 and 
                pd.api.types.is_numeric_dtype(self.X_train[feature])):
                self.continuous_features.append(feature)
            else:
                self.discrete_features.append(feature)
    
    def calculate_gini(self, y):
        """
        计算基尼指数
        参数: y - 标签序列
        返回: 基尼指数
        """
        if len(y) == 0:
            return 0
        
        # 统计每个类别的数量
        counts = np.bincount(y) if y.dtype == np.int64 else np.array(list(Counter(y).values()))
        probabilities = counts / len(y)
        
        # 计算基尼指数: 1 - Σ(p_i^2)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def calculate_gini_index(self, X, y, feature):
        """
        计算某个特征的基尼指数
        参数:
            X - 特征数据
            y - 标签数据
            feature - 特征名称
        返回: (最小基尼指数, 最佳划分点)
        """
        if feature in self.discrete_features:
            return self._discrete_feature_gini_index(X, y, feature)
        else:
            return self._continuous_feature_gini_index(X, y, feature)
    
    def _discrete_feature_gini_index(self, X, y, feature):
        """计算离散特征的基尼指数（二分法）"""
        feature_values = X[feature].unique()
        best_gini = float('inf')
        best_split = None
        
        # 对于离散特征，尝试所有可能的二分划分
        for i in range(1, len(feature_values)):
            # 生成所有可能的二分组合
            for subset in self._generate_binary_splits(feature_values, i):
                left_mask = X[feature].isin(subset)
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # 计算加权基尼指数
                left_weight = len(y_left) / len(y)
                right_weight = len(y_right) / len(y)
                weighted_gini = (left_weight * self.calculate_gini(y_left) + 
                               right_weight * self.calculate_gini(y_right))
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = (subset, list(set(feature_values) - set(subset)))
        
        return best_gini, best_split
    
    def _continuous_feature_gini_index(self, X, y, feature):
        """计算连续特征的基尼指数（通过最佳划分点）"""
        # 对特征值排序
        sorted_indices = X[feature].sort_values().index
        unique_values = sorted(X[feature].unique())
        
        if len(unique_values) <= 1:
            return float('inf'), None
        
        best_gini = float('inf')
        best_split_point = None
        
        # 尝试所有可能的划分点（相邻值的中间点）
        for i in range(len(unique_values) - 1):
            split_point = (unique_values[i] + unique_values[i + 1]) / 2
            
            # 根据划分点划分数据
            left_mask = X[feature] <= split_point
            right_mask = X[feature] > split_point
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            # 计算加权基尼指数
            left_weight = len(y_left) / len(y)
            right_weight = len(y_right) / len(y)
            weighted_gini = (left_weight * self.calculate_gini(y_left) + 
                           right_weight * self.calculate_gini(y_right))
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_split_point = split_point
        
        return best_gini, best_split_point
    
    def _generate_binary_splits(self, values, subset_size):
        """生成离散特征的所有可能二分划分"""
        from itertools import combinations
        return list(combinations(values, subset_size))
    
    def choose_best_split(self, X, y, features):
        """选择基尼指数最小的特征和划分点"""
        best_gini = float('inf')
        best_feature = None
        best_split = None
        
        for feature in features:
            gini, split = self.calculate_gini_index(X, y, feature)
            
            if feature in self.discrete_features:
                print(f"离散特征 '{feature}' 的最小基尼指数: {gini:.4f}")
            else:
                print(f"连续特征 '{feature}' 的最小基尼指数: {gini:.4f}")
            
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_split = split
        
        if best_feature in self.continuous_features:
            print(f"选择最佳划分特征: {best_feature}, 基尼指数: {best_gini:.4f}, 划分点: {best_split}")
        else:
            print(f"选择最佳划分特征: {best_feature}, 基尼指数: {best_gini:.4f}")
        
        return best_feature, best_split
    
    def build_tree(self, X, y, features, depth=0):
        """递归构建CART决策树"""
        # 终止条件检查
        if self._should_stop(y, features, depth):
            return self._get_leaf_value(y)
        
        # 选择最佳划分
        best_feature, best_split = self.choose_best_split(X, y, features)
        
        if best_feature is None:
            return self._get_leaf_value(y)
        
        # 创建树节点
        tree = {'feature': best_feature, 'split': best_split, 'children': {}}
        
        # 从特征列表中移除已选特征（对于离散特征）
        remaining_features = [f for f in features if f != best_feature]
        
        # 根据特征类型进行划分
        if best_feature in self.discrete_features:
            # 离散特征：二分划分
            left_values, right_values = best_split
            left_mask = X[best_feature].isin(left_values)
            right_mask = X[best_feature].isin(right_values)
            
            # 左子树
            X_left = X[left_mask]
            y_left = y[left_mask]
            if len(y_left) == 0:
                tree['children']['left'] = self._get_leaf_value(y)
            else:
                tree['children']['left'] = self.build_tree(
                    X_left, y_left, remaining_features, depth + 1
                )
            
            # 右子树
            X_right = X[right_mask]
            y_right = y[right_mask]
            if len(y_right) == 0:
                tree['children']['right'] = self._get_leaf_value(y)
            else:
                tree['children']['right'] = self.build_tree(
                    X_right, y_right, remaining_features, depth + 1
                )
        else:
            # 连续特征：按划分点二分
            left_mask = X[best_feature] <= best_split
            right_mask = X[best_feature] > best_split
            
            # 左子树
            X_left = X[left_mask]
            y_left = y[left_mask]
            if len(y_left) == 0:
                tree['children']['left'] = self._get_leaf_value(y)
            else:
                tree['children']['left'] = self.build_tree(
                    X_left, y_left, remaining_features, depth + 1
                )
            
            # 右子树
            X_right = X[right_mask]
            y_right = y[right_mask]
            if len(y_right) == 0:
                tree['children']['right'] = self._get_leaf_value(y)
            else:
                tree['children']['right'] = self.build_tree(
                    X_right, y_right, remaining_features, depth + 1
                )
        
        return tree
    
    def _should_stop(self, y, features, depth):
        """检查是否应该停止分裂"""
        # 所有样本属于同一类别
        if len(np.unique(y)) == 1:
            return True
        
        # 没有特征可用
        if len(features) == 0:
            return True
        
        # 样本数小于最小分裂样本数
        if len(y) < self.min_samples_split:
            return True
        
        # 达到最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        
        return False
    
    def _get_leaf_value(self, y):
        """获取叶节点的值（多数类）"""
        return y.mode()[0] if len(y) > 0 else None
    
    def fit(self, X_train, y_train):
        """训练CART决策树模型"""
        print("开始构建CART决策树...")
        self.feature_names = list(X_train.columns)
        self._identify_feature_types()
        self.tree = self.build_tree(X_train, y_train, self.feature_names)
        print("CART决策树构建完成!")
        return self
    
    def predict_sample(self, sample, tree):
        """对单个样本进行预测"""
        if not isinstance(tree, dict):
            return tree
        
        feature = tree['feature']
        split = tree['split']
        
        if feature in self.discrete_features:
            # 离散特征
            feature_value = sample[feature]
            if feature_value in split[0]:  # 属于左子树的值集合
                return self.predict_sample(sample, tree['children']['left'])
            else:  # 属于右子树的值集合
                return self.predict_sample(sample, tree['children']['right'])
        else:
            # 连续特征
            feature_value = sample[feature]
            if feature_value <= split:  # 划分点
                return self.predict_sample(sample, tree['children']['left'])
            else:
                return self.predict_sample(sample, tree['children']['right'])
    
    def predict(self, X_test):
        """对测试集进行预测"""
        if self.tree is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        predictions = []
        for i in range(len(X_test)):
            sample = X_test.iloc[i]
            prediction = self.predict_sample(sample, self.tree)
            predictions.append(prediction)
        
        return predictions
    
    def calculate_accuracy(self, y_true, y_pred):
        """计算分类准确率"""
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

def main_cart():
    """CART决策树主函数"""
    # 创建CART决策树实例
    cart_tree = CARTDecisionTree(min_samples_split=2, max_depth=5)
    
    # 加载包含连续属性的数据
    X_train, y_train, X_test, y_test = cart_tree.load_data(
        "watermelon-train2.csv", 
        "watermelon-test2.csv"
    )
    
    # 训练模型
    cart_tree.fit(X_train, y_train)
    
    # 进行预测
    y_pred = cart_tree.predict(X_test)
    
    # 计算准确率
    accuracy = cart_tree.calculate_accuracy(y_test, y_pred)
    
    print(f"\n在测试集watermelon-test2上的分类准确率: {accuracy:.2f}")
    
    # 输出预测结果对比
    print("\n预测结果对比:")
    print("真实标签:", list(y_test))
    print("预测标签:", y_pred)

if __name__ == "__main__":
    main_cart()