import pandas as pd
import numpy as np
import math
from collections import Counter

class ID3DecisionTreeWithPruning:
    def __init__(self, pruning_method='pep', confidence_level=0.25):
        self.tree = None
        self.feature_names = None
        self.pruning_method = pruning_method  # 'pep', 'rep', 'mep'
        self.confidence_level = confidence_level  # 用于PEP剪枝的置信度
    
    def load_data(self, train_file, test_file):
        """加载训练集和测试集数据"""
        train_data = pd.read_csv(train_file, encoding='gbk')
        test_data = pd.read_csv(test_file, encoding='gbk')
        
        self.X_train = train_data.iloc[:, 1:-1]
        self.y_train = train_data.iloc[:, -1]
        self.X_test = test_data.iloc[:, 1:-1]
        self.y_test = test_data.iloc[:, -1]
        
        self.feature_names = list(self.X_train.columns)
        
        print("训练集形状:", self.X_train.shape)
        print("测试集形状:", self.X_test.shape)
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def calculate_entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y) if y.dtype == np.int64 else np.array(list(Counter(y).values()))
        probabilities = counts / len(y)
        entropy = -np.sum([p * math.log2(p) for p in probabilities if p > 0])
        return entropy
    
    def calculate_information_gain(self, X, y, feature):
        """计算信息增益"""
        total_entropy = self.calculate_entropy(y)
        feature_values = X[feature].unique()
        weighted_entropy = 0.0
        total_samples = len(y)
        
        for value in feature_values:
            subset_mask = X[feature] == value
            y_subset = y[subset_mask]
            if len(y_subset) > 0:
                subset_entropy = self.calculate_entropy(y_subset)
                subset_weight = len(y_subset) / total_samples
                weighted_entropy += subset_weight * subset_entropy
        
        return total_entropy - weighted_entropy
    
    def choose_best_feature(self, X, y, features):
        """选择最佳划分特征"""
        best_gain = -1
        best_feature = None
        
        for feature in features:
            gain = self.calculate_information_gain(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        return best_feature
    
    def build_tree_with_samples(self, X, y, features, depth=0):
        """
        构建决策树并记录每个节点的样本信息（用于剪枝）
        返回：包含样本信息的决策树
        """
        # 记录当前节点的样本信息
        node_info = {
            'samples': len(y),
            'majority_class': y.mode()[0] if len(y) > 0 else None,
            'class_distribution': dict(Counter(y))
        }
        
        # 终止条件1: 所有样本属于同一类别
        if len(np.unique(y)) == 1:
            node_info['leaf'] = True
            node_info['label'] = y.iloc[0]
            return node_info
        
        # 终止条件2: 没有特征可用
        if len(features) == 0:
            node_info['leaf'] = True
            node_info['label'] = y.mode()[0]
            return node_info
        
        # 选择最佳划分特征
        best_feature = self.choose_best_feature(X, y, features)
        if best_feature is None:
            node_info['leaf'] = True
            node_info['label'] = y.mode()[0]
            return node_info
        
        node_info['leaf'] = False
        node_info['feature'] = best_feature
        node_info['children'] = {}
        
        # 从特征列表中移除已选特征
        remaining_features = [f for f in features if f != best_feature]
        
        # 递归构建子树
        for value in X[best_feature].unique():
            subset_mask = X[best_feature] == value
            X_subset = X[subset_mask]
            y_subset = y[subset_mask]
            
            if len(y_subset) == 0:
                # 子集为空，创建叶节点
                child_node = {
                    'leaf': True,
                    'label': y.mode()[0],
                    'samples': 0,
                    'majority_class': y.mode()[0],
                    'class_distribution': dict(Counter(y))
                }
            else:
                # 递归构建子树
                child_node = self.build_tree_with_samples(X_subset, y_subset, remaining_features, depth + 1)
            
            node_info['children'][value] = child_node
        
        return node_info
    
    def calculate_node_error(self, node):
        """计算节点的错误率"""
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
    
    def pessimistic_error_pruning(self, node):
        """
        悲观错误剪枝(PEP)
        基于二项分布和连续性校正的剪枝方法
        """
        if node['leaf']:
            return node
        
        # 计算当前节点作为叶节点的悲观错误率
        if node['samples'] == 0:
            return node
        
        # 叶节点的错误率计算（带连续性校正）
        leaf_error_count = node['samples'] - node['class_distribution'].get(node['majority_class'], 0)
        leaf_pessimistic_error = (leaf_error_count + 0.5) / node['samples']
        
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
            print(f"PEP剪枝: 将特征 '{node.get('feature', '')}' 节点剪枝为叶节点，类别 '{node['label']}'")
        
        else:
            # 递归剪枝子树
            for value, child_node in node['children'].items():
                node['children'][value] = self.pessimistic_error_pruning(child_node)
        
        return node
    
    def get_leaf_nodes(self, node):
        """获取决策树中的所有叶节点"""
        leaf_nodes = []
        if node['leaf']:
            leaf_nodes.append(node)
        else:
            for child in node['children'].values():
                leaf_nodes.extend(self.get_leaf_nodes(child))
        return leaf_nodes
    
    def prune_tree(self, tree):
        """执行剪枝"""
        print("开始剪枝...")
        if self.pruning_method == 'pep':
            return self.pessimistic_error_pruning(tree)
        else:
            # 可以扩展其他剪枝方法
            return tree
    
    def fit(self, X_train, y_train, prune=True):
        """训练决策树并进行剪枝"""
        print("开始构建ID3决策树...")
        self.feature_names = list(X_train.columns)
        
        # 构建包含样本信息的决策树
        self.tree = self.build_tree_with_samples(X_train, y_train, self.feature_names)
        print("决策树构建完成!")
        
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
        feature_value = sample[feature]
        
        if feature_value in tree['children']:
            return self.predict_sample(sample, tree['children'][feature_value])
        else:
            # 如果遇到未知特征值，返回父节点的多数类
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
        return correct / len(y_true)

def main_id3_with_pruning():
    """带剪枝的ID3决策树主函数"""
    print("=" * 60)
    print("ID3决策树剪枝实验")
    print("=" * 60)
    
    # 创建ID3决策树实例（带剪枝）
    id3_tree_pruned = ID3DecisionTreeWithPruning(pruning_method='pep', confidence_level=0.25)
    
    # 加载数据
    X_train, y_train, X_test, y_test = id3_tree_pruned.load_data(
        "watermelon-train1.csv", 
        "watermelon-test1.csv"
    )
    
    print("\n1. 剪枝前的决策树:")
    # 创建不剪枝的决策树用于对比
    id3_tree_no_prune = ID3DecisionTreeWithPruning(pruning_method='pep')
    id3_tree_no_prune.fit(X_train, y_train, prune=False)
    y_pred_no_prune = id3_tree_no_prune.predict(X_test)
    accuracy_no_prune = id3_tree_no_prune.calculate_accuracy(y_test, y_pred_no_prune)
    
    print("\n2. 剪枝后的决策树:")
    # 训练带剪枝的决策树
    id3_tree_pruned.fit(X_train, y_train, prune=True)
    y_pred_pruned = id3_tree_pruned.predict(X_test)
    accuracy_pruned = id3_tree_pruned.calculate_accuracy(y_test, y_pred_pruned)
    
    # 结果对比
    print("\n" + "=" * 60)
    print("剪枝效果对比分析")
    print("=" * 60)
    print(f"剪枝前准确率: {accuracy_no_prune:.4f}")
    print(f"剪枝后准确率: {accuracy_pruned:.4f}")
    print(f"准确率变化: {accuracy_pruned - accuracy_no_prune:+.4f}")
    
    # 树复杂度对比
    leaves_no_prune = len(id3_tree_no_prune.get_leaf_nodes(id3_tree_no_prune.tree))
    leaves_pruned = len(id3_tree_pruned.get_leaf_nodes(id3_tree_pruned.tree))
    depth_no_prune = id3_tree_no_prune.calculate_tree_depth(id3_tree_no_prune.tree)
    depth_pruned = id3_tree_pruned.calculate_tree_depth(id3_tree_pruned.tree)
    
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
    
    return id3_tree_no_prune, id3_tree_pruned, accuracy_no_prune, accuracy_pruned

if __name__ == "__main__":
    # 运行ID3剪枝实验
    tree_no_prune, tree_pruned, acc_no_prune, acc_pruned = main_id3_with_pruning()
