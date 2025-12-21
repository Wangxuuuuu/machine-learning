import pandas as pd
import numpy as np
import math
from collections import Counter

class C45DecisionTreeWithPruning:
    def __init__(self, min_samples_split=2, max_depth=None, pruning_method='pep', confidence_level=0.25):
        self.tree = None
        self.feature_names = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.pruning_method = pruning_method
        self.confidence_level = confidence_level
        self.continuous_features = []
        self.discrete_features = []
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
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
        if self.X_train is None:
            return
            
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
        # 处理字符串标签
        if isinstance(y, pd.Series) and y.dtype == 'object':
            counts = np.array(list(Counter(y).values()))
        else:
            counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = -np.sum([p * math.log2(p) for p in probabilities if p > 0])
        return entropy
    
    def calculate_information_gain_ratio(self, X, y, feature):
        """计算信息增益率"""
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
                if probability > 0:
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
        unique_values = sorted(X[feature].unique())
        
        if len(unique_values) <= 1:
            return 0, None
        
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
            if left_weight > 0 and right_weight > 0:
                split_info = - (left_weight * math.log2(left_weight) + 
                              right_weight * math.log2(right_weight))
            else:
                split_info = 0
            
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
                if gain_ratio is not None:
                    print(f"连续特征 '{feature}' 的信息增益率: {gain_ratio:.4f}, 最佳划分点: {split_point}")
                    
                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_feature = feature
                        best_split_point = split_point
        
        if best_feature is not None:
            print(f"选择最佳划分特征: {best_feature}, 信息增益率: {best_gain_ratio:.4f}")
            if best_split_point is not None:
                print(f"最佳划分点: {best_split_point}")
        
        return best_feature, best_split_point
    
    def build_tree_with_samples(self, X, y, features, depth=0):
        """构建包含样本信息的C4.5决策树（用于剪枝）"""
        # 记录当前节点的样本信息
        node_info = {
            'samples': len(y),
            'majority_class': y.mode()[0] if len(y) > 0 else None,
            'class_distribution': dict(Counter(y)),
            'depth': depth
        }
        
        # 终止条件检查
        if self._should_stop(y, features, depth):
            node_info['leaf'] = True
            node_info['label'] = self._get_leaf_value(y)
            return node_info
        
        # 选择最佳特征
        best_feature, split_point = self.choose_best_feature(X, y, features)
        
        if best_feature is None:
            node_info['leaf'] = True
            node_info['label'] = self._get_leaf_value(y)
            return node_info
        
        node_info['leaf'] = False
        node_info['feature'] = best_feature
        node_info['split_point'] = split_point
        node_info['children'] = {}
        
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
                    # 子集为空，创建叶节点
                    child_node = {
                        'leaf': True,
                        'label': self._get_leaf_value(y),
                        'samples': 0,
                        'majority_class': self._get_leaf_value(y),
                        'class_distribution': dict(Counter(y)),
                        'depth': depth + 1
                    }
                else:
                    # 递归构建子树
                    child_node = self.build_tree_with_samples(
                        X_subset, y_subset, remaining_features, depth + 1
                    )
                
                node_info['children'][value] = child_node
        else:
            # 连续特征：按划分点二分
            if split_point is None:
                # 如果没有找到合适的划分点，创建叶节点
                node_info['leaf'] = True
                node_info['label'] = self._get_leaf_value(y)
                return node_info
                
            left_mask = X[best_feature] <= split_point
            right_mask = X[best_feature] > split_point
            
            # 左子树（<= 划分点）
            X_left = X[left_mask]
            y_left = y[left_mask]
            if len(y_left) == 0:
                node_info['children']['left'] = {
                    'leaf': True,
                    'label': self._get_leaf_value(y),
                    'samples': 0,
                    'majority_class': self._get_leaf_value(y),
                    'class_distribution': dict(Counter(y)),
                    'depth': depth + 1
                }
            else:
                node_info['children']['left'] = self.build_tree_with_samples(
                    X_left, y_left, remaining_features, depth + 1
                )
            
            # 右子树（> 划分点）
            X_right = X[right_mask]
            y_right = y[right_mask]
            if len(y_right) == 0:
                node_info['children']['right'] = {
                    'leaf': True,
                    'label': self._get_leaf_value(y),
                    'samples': 0,
                    'majority_class': self._get_leaf_value(y),
                    'class_distribution': dict(Counter(y)),
                    'depth': depth + 1
                }
            else:
                node_info['children']['right'] = self.build_tree_with_samples(
                    X_right, y_right, remaining_features, depth + 1
                )
        
        return node_info
    
    def _should_stop(self, y, features, depth):
        """检查是否应该停止分裂"""
        if len(y) == 0:
            return True
        if len(np.unique(y)) == 1:
            return True
        if len(features) == 0:
            return True
        if len(y) < self.min_samples_split:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False
    
    def _get_leaf_value(self, y):
        """获取叶节点的值（多数类）"""
        if len(y) == 0:
            return None
        return y.mode()[0]
    
    def calculate_node_error(self, node):
        """计算节点的错误数"""
        if node['leaf']:
            # 叶节点的错误数
            total_samples = node['samples']
            if total_samples == 0:
                return 0, 0
            majority_class_count = node['class_distribution'].get(node['label'], 0)
            error_count = total_samples - majority_class_count
            return error_count, total_samples
        else:
            # 内部节点的错误数是子节点错误数之和
            total_error = 0
            total_samples = 0
            for child in node['children'].values():
                child_error, child_samples = self.calculate_node_error(child)
                total_error += child_error
                total_samples += child_samples
            return total_error, total_samples
    
    def get_leaf_nodes(self, node):
        """获取决策树中的所有叶节点"""
        leaf_nodes = []
        if node['leaf']:
            leaf_nodes.append(node)
        else:
            for child in node['children'].values():
                leaf_nodes.extend(self.get_leaf_nodes(child))
        return leaf_nodes
    
    def pessimistic_error_pruning(self, node):
        """悲观错误剪枝(PEP)"""
        if node['leaf']:
            return node
        
        # 计算当前节点作为叶节点的悲观错误率
        if node['samples'] == 0:
            return node
        
        # 叶节点的错误率计算（带连续性校正）
        leaf_error_count = node['samples'] - node['class_distribution'].get(node['majority_class'], 0)
        leaf_pessimistic_error = (leaf_error_count + 0.5) / node['samples'] if node['samples'] > 0 else 0
        
        # 计算子树的总错误率
        subtree_error_count, subtree_samples = self.calculate_node_error(node)
        if subtree_samples == 0:
            return node
        
        # 子树的悲观错误率（带连续性校正）
        subtree_pessimistic_error = (subtree_error_count + 0.5 * len(self.get_leaf_nodes(node))) / subtree_samples
        
        # 如果叶节点的悲观错误率小于等于子树的悲观错误率，则剪枝
        if leaf_pessimistic_error <= subtree_pessimistic_error:
            # 剪枝：将当前节点变为叶节点
            node['leaf'] = True
            node['label'] = node['majority_class']
            node['children'] = {}  # 删除子树
            feature_name = node.get('feature', '')
            if feature_name:
                print(f"PEP剪枝: 将特征 '{feature_name}' 节点剪枝为叶节点，类别 '{node['label']}'")
        
        else:
            # 递归剪枝子树
            for key, child_node in list(node['children'].items()):
                node['children'][key] = self.pessimistic_error_pruning(child_node)
        
        return node
    
    def prune_tree(self, tree):
        """执行剪枝"""
        print("开始剪枝...")
        if self.pruning_method == 'pep':
            return self.pessimistic_error_pruning(tree)
        else:
            # 可以扩展其他剪枝方法
            return tree
    
    def fit(self, X_train, y_train, prune=True):
        """训练C4.5决策树模型并进行剪枝"""
        print("开始构建C4.5决策树...")
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = list(X_train.columns)
        
        # 识别特征类型
        self._identify_feature_types()
        
        # 构建包含样本信息的决策树
        self.tree = self.build_tree_with_samples(X_train, y_train, self.feature_names)
        print("C4.5决策树构建完成!")
        
        # 计算剪枝前的树复杂度
        pre_prune_leaves = len(self.get_leaf_nodes(self.tree))
        pre_prune_depth = self.calculate_tree_depth(self.tree)
        print(f"剪枝前 - 叶节点数: {pre_prune_leaves}, 树深度: {pre_prune_depth}")
        
        if prune:
            # 执行剪枝
            self.tree = self.prune_tree(self.tree)
            
            # 计算剪枝后的树复杂度
            post_prune_leaves = len(self.get_leaf_nodes(self.tree))
            post_prune_depth = self.calculate_tree_depth(self.tree)
            print(f"剪枝后 - 叶节点数: {post_prune_leaves}, 树深度: {post_prune_depth}")
            print(f"剪枝效果: 叶节点减少 {pre_prune_leaves - post_prune_leaves} 个")
        
        return self
    
    def calculate_tree_depth(self, node):
        """计算决策树的深度"""
        if node['leaf']:
            return 1
        else:
            max_depth = 0
            for child in node['children'].values():
                child_depth = self.calculate_tree_depth(child)
                if child_depth > max_depth:
                    max_depth = child_depth
            return max_depth + 1
    
    def predict_sample(self, sample, tree):
        """对单个样本进行预测"""
        if tree['leaf']:
            return tree['label']
        
        feature = tree['feature']
        
        if feature in self.discrete_features:
            # 离散特征
            feature_value = sample[feature]
            if feature_value in tree['children']:
                return self.predict_sample(sample, tree['children'][feature_value])
            else:
                # 处理未知特征值
                return tree['majority_class']
        else:
            # 连续特征
            feature_value = sample[feature]
            split_point = tree['split_point']
            if split_point is None:
                return tree['majority_class']
                
            if feature_value <= split_point:
                if 'left' in tree['children']:
                    return self.predict_sample(sample, tree['children']['left'])
                else:
                    return tree['majority_class']
            else:
                if 'right' in tree['children']:
                    return self.predict_sample(sample, tree['children']['right'])
                else:
                    return tree['majority_class']
    
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
        return correct / len(y_true) if len(y_true) > 0 else 0

def main_c45_with_pruning():
    """带剪枝的C4.5决策树主函数"""
    print("=" * 60)
    print("C4.5决策树剪枝实验")
    print("=" * 60)
    
    # 创建C4.5决策树实例（带剪枝）
    c45_tree_pruned = C45DecisionTreeWithPruning(
        min_samples_split=2, 
        max_depth=5, 
        pruning_method='pep', 
        confidence_level=0.25
    )
    
    # 加载包含连续属性的数据
    X_train, y_train, X_test, y_test = c45_tree_pruned.load_data(
        "watermelon-train2.csv", 
        "watermelon-test2.csv"
    )
    
    print("\n1. 剪枝前的决策树:")
    # 创建不剪枝的决策树用于对比
    c45_tree_no_prune = C45DecisionTreeWithPruning(min_samples_split=2, max_depth=5)
    c45_tree_no_prune.fit(X_train, y_train, prune=False)
    y_pred_no_prune = c45_tree_no_prune.predict(X_test)
    accuracy_no_prune = c45_tree_no_prune.calculate_accuracy(y_test, y_pred_no_prune)
    
    print("\n2. 剪枝后的决策树:")
    # 训练带剪枝的决策树
    c45_tree_pruned.fit(X_train, y_train, prune=True)
    y_pred_pruned = c45_tree_pruned.predict(X_test)
    accuracy_pruned = c45_tree_pruned.calculate_accuracy(y_test, y_pred_pruned)
    
    # 结果对比
    print("\n" + "=" * 60)
    print("剪枝效果对比分析")
    print("=" * 60)
    print(f"剪枝前准确率: {accuracy_no_prune:.4f}")
    print(f"剪枝后准确率: {accuracy_pruned:.4f}")
    print(f"准确率变化: {accuracy_pruned - accuracy_no_prune:+.4f}")
    
    # 树复杂度对比
    leaves_no_prune = len(c45_tree_no_prune.get_leaf_nodes(c45_tree_no_prune.tree))
    leaves_pruned = len(c45_tree_pruned.get_leaf_nodes(c45_tree_pruned.tree))
    depth_no_prune = c45_tree_no_prune.calculate_tree_depth(c45_tree_no_prune.tree)
    depth_pruned = c45_tree_pruned.calculate_tree_depth(c45_tree_pruned.tree)
    
    print(f"\n树复杂度对比:")
    print(f"剪枝前 - 叶节点数: {leaves_no_prune}, 树深度: {depth_no_prune}")
    print(f"剪枝后 - 叶节点数: {leaves_pruned}, 树深度: {depth_pruned}")
    print(f"叶节点减少: {leaves_no_prune - leaves_pruned}")
    
    # 分析剪枝对泛化能力的影响
    print("\n剪枝对模型泛化能力的影响分析:")
    if accuracy_pruned > accuracy_no_prune:
        print("✓ 剪枝提高了模型泛化能力，减少了过拟合")
    elif accuracy_pruned == accuracy_no_prune:
        print("○ 剪枝保持了模型泛化能力，但简化了模型结构")
    else:
        print("✗ 剪枝可能过度简化了模型，导致欠拟合")
    
    return c45_tree_no_prune, c45_tree_pruned, accuracy_no_prune, accuracy_pruned

if __name__ == "__main__":
    # 运行C4.5剪枝实验
    tree_no_prune, tree_pruned, acc_no_prune, acc_pruned = main_c45_with_pruning()