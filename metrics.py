import numpy as np

# ==========================================
# 第一部分：compute_accuracy
# ==========================================
def compute_accuracy(predictions, ground_truth):
    """
    计算准确率
    predictions: 模型预测的标签列表，如 [0, 1, 0, 0, 1]
    ground_truth: 真实标签列表，如 [0, 1, 1, 0, 0]
    """
    preds = np.array(predictions)
    truths = np.array(ground_truth)
    correct = (preds == truths).sum()
    total = len(truths)
    return correct / total


# ==========================================
# 第二部分：sample_data
# ==========================================
def sample_data(images, labels, ratio=0.1):
    """
    从完整数据集中随机抽取 ratio 比例的数据
    images: 所有图片数据，numpy数组，形状为 [N, 32, 32, 3]
    labels: 所有标签，numpy数组，形状为 [N]
    ratio: 抽取比例，默认0.1（即10%）
    返回: 抽取后的图片和标签
    """
    num_samples = len(images)#获取图片总数
    sample_count = int(num_samples * ratio)#计算要抽取多少张
    indices = np.random.choice(num_samples, sample_count, replace=False)#replace=False 表示不重复抽取。
    sampled_images = images[indices]
    sampled_labels = labels[indices]
    return sampled_images, sampled_labels


# ==========================================
# 第三部分：compute_all_metrics计算所有指标
# ==========================================
def compute_all_metrics(clean_acc, attack_acc, all_attack_accs=None):
    """
    计算所有评价指标
    clean_acc: 干净样本上的准确率
    attack_acc: 攻击样本上的准确率
    all_attack_accs: 多个攻击强度下的准确率列表，用于算最差性能
    """
    retention = attack_acc / clean_acc if clean_acc > 0 else 0#计算性能保持率
    degradation = 1 - retention#计算退化幅度
    worst_case = min(all_attack_accs) if all_attack_accs else attack_acc#算最差性能

    return {
        "clean_accuracy": clean_acc,
        "attack_accuracy": attack_acc,
        "performance_retention": retention,
        "performance_degradation": degradation,
        "worst_case_accuracy": worst_case,
    }


# ==========================================
# 第四部分：用假数据测试所有函数
# ==========================================
if __name__ == "__main__":
    print("=" * 40)
    print("测试 metrics.py 所有函数（假数据）")
    print("=" * 40)

    # 测试 compute_accuracy
    print("\n[测试1] compute_accuracy")
    fake_preds = [0, 1, 1, 0, 0]
    fake_labels = [0, 1, 0, 0, 0]
    acc = compute_accuracy(fake_preds, fake_labels)
    print(f"  准确率: {acc} (期望: 0.8)")

    # 测试 sample_data
    print("\n[测试2] sample_data")
    fake_images = np.random.randn(100, 32, 32, 3)#生成 100 张假的"图片"（实际上是随机数字），形状是 [100, 32, 32, 3]，表示 100 张 32x32 像素的彩色图。
    fake_labels = np.random.randint(0, 10, size=100)#生成 100 个随机标签（0-9 之间的整数），模拟每张图片对应的类别。
    sampled_imgs, sampled_lbls = sample_data(fake_images, fake_labels, 0.1)
    print(f"  原始: 100张 → 采样后: {len(sampled_imgs)}张 (期望: 10张)")

    # 测试 compute_all_metrics
    print("\n[测试3] compute_all_metrics")
    clean_acc = 0.92
    attack_acc = 0.78
    all_accs = [0.78, 0.75, 0.72]
    metrics = compute_all_metrics(clean_acc, attack_acc, all_accs)
    print(f"  性能保持率: {metrics['performance_retention']:.3f}")
    print(f"  退化幅度: {metrics['performance_degradation']:.3f}")
    print(f"  最差性能: {metrics['worst_case_accuracy']:.3f}")

    print("\n" + "=" * 40)
    print("所有测试通过！✅")
    print("=" * 40)