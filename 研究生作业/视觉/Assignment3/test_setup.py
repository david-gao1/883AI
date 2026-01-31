"""
测试脚本
用于验证环境和代码是否能正常运行
"""

import os
import sys

def test_imports():
    """测试必要的库是否已安装"""
    print("测试库导入...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch 未安装")
        return False
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError:
        print("✗ torchvision 未安装")
        return False
    
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
    except ImportError:
        print("✗ numpy 未安装")
        return False
    
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError:
        print("✗ matplotlib 未安装")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        print("✗ scikit-learn 未安装")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow")
    except ImportError:
        print("✗ Pillow 未安装")
        return False
    
    try:
        import tqdm
        print("✓ tqdm")
    except ImportError:
        print("✗ tqdm 未安装")
        return False
    
    return True


def test_data():
    """测试数据文件是否存在"""
    print("\n测试数据文件...")
    
    raw_data_dir = './raw_data'
    groundtruth_dir = './groundtruth'
    
    if not os.path.exists(raw_data_dir):
        print(f"✗ 原始数据目录不存在: {raw_data_dir}")
        return False
    else:
        print(f"✓ 原始数据目录存在: {raw_data_dir}")
    
    if not os.path.exists(groundtruth_dir):
        print(f"✗ 标注数据目录不存在: {groundtruth_dir}")
        return False
    else:
        print(f"✓ 标注数据目录存在: {groundtruth_dir}")
    
    # 检查是否有数据文件
    raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.jpg')]
    gt_files = [f for f in os.listdir(groundtruth_dir) if f.endswith('.png')]
    
    print(f"  找到 {len(raw_files)} 张原始图像")
    print(f"  找到 {len(gt_files)} 张标注图像")
    
    if len(raw_files) == 0:
        print("✗ 未找到原始图像文件")
        return False
    
    return True


def test_model():
    """测试模型是否能正常创建"""
    print("\n测试模型创建...")
    try:
        from model import get_model
        from data_loader import NUM_CLASSES
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  使用设备: {device}")
        
        model = get_model(model_name='unet', n_channels=3, n_classes=NUM_CLASSES)
        model = model.to(device)
        
        # 测试前向传播
        test_input = torch.randn(1, 3, 256, 256).to(device)
        output = model(test_input)
        
        print(f"✓ 模型创建成功")
        print(f"  输入形状: {test_input.shape}")
        print(f"  输出形状: {output.shape}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  总参数数: {total_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False


def test_data_loader():
    """测试数据加载器"""
    print("\n测试数据加载器...")
    try:
        from data_loader import get_data_loaders
        
        train_ids = list(range(1, 11))  # 只测试前10张
        test_ids = list(range(201, 211))  # 只测试前10张
        
        train_loader, test_loader = get_data_loaders(
            './raw_data',
            './groundtruth',
            train_ids,
            test_ids,
            batch_size=2
        )
        
        print(f"✓ 数据加载器创建成功")
        print(f"  训练集批次: {len(train_loader)}")
        print(f"  测试集批次: {len(test_loader)}")
        
        # 测试加载一个批次
        for images, masks in train_loader:
            print(f"  图像形状: {images.shape}")
            print(f"  标注形状: {masks.shape}")
            break
        
        return True
    except Exception as e:
        print(f"✗ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 50)
    print("环境测试")
    print("=" * 50)
    
    results = []
    
    # 测试导入
    results.append(("库导入", test_imports()))
    
    # 测试数据
    results.append(("数据文件", test_data()))
    
    # 测试模型
    if results[0][1]:  # 如果导入成功
        results.append(("模型创建", test_model()))
    
    # 测试数据加载器
    if results[0][1] and results[1][1]:  # 如果导入和数据都成功
        results.append(("数据加载器", test_data_loader()))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("所有测试通过！可以开始训练模型。")
        print("\n运行训练:")
        print("  python main.py --mode train")
    else:
        print("部分测试失败，请检查环境配置。")
        print("\n安装依赖:")
        print("  pip install -r requirements.txt")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
