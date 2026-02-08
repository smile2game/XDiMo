import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torchvision import transforms
import os
import glob
import time
import logging
from tqdm import tqdm
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fvd_calculation.log')
    ]
)
logger = logging.getLogger(__name__)

# I3D 模型路径（共享权重放在 ckpts/shared/）
_ROOT = os.path.dirname(os.path.abspath(__file__))
I3D_MODEL_PATH = os.path.join(_ROOT, 'ckpts', 'shared', 'i3d_torchscript.pt')

# 定义 I3D 模型类
class I3D(nn.Module):
    def __init__(self, num_classes=400):
        super(I3D, self).__init__()
        try:
            # 加载预训练 I3D 模型
            self.model = torch.jit.load(I3D_MODEL_PATH)
            self.model.eval()
            logger.info(f"成功加载 I3D 模型: {I3D_MODEL_PATH}")
        except Exception as e:
            logger.error(f"加载 I3D 模型失败: {str(e)}")
            raise

    def forward(self, x):
        return self.model(x)

# 从视频中提取帧
def extract_frames(video_path, num_frames=16):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    frames = []
    if total_frames == 0:
        raise ValueError(f"视频 {video_path} 没有可用的帧")
    
    # 记录视频信息
    video_info = {
        'path': video_path,
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration,
        'sampled_frames': min(num_frames, total_frames)
    }
    
    # 均匀采样 num_frames 个帧
    step = max(1, total_frames // num_frames)
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB
        frame = cv2.resize(frame, (224, 224))  # 调整为 224x224
        frames.append(frame)
        if len(frames) >= num_frames:
            break
    
    cap.release()
    
    # 如果帧数不足，填充最后一帧
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    processing_time = time.time() - start_time
    video_info['processing_time'] = processing_time
    video_info['actual_sampled'] = len(frames)
    
    logger.debug(f"视频处理完成: {os.path.basename(video_path)} - 总帧数: {total_frames} - 采样帧数: {len(frames)} - 耗时: {processing_time:.2f}s")
    
    return np.array(frames[:num_frames]), video_info

# 预处理帧
def preprocess_frames(frames):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])
    # 将帧转换为 tensor 并调整维度为 (C, T, H, W)
    return torch.stack([transform(frame) for frame in frames]).permute(1, 0, 2, 3).unsqueeze(0)

# 使用 I3D 提取特征
def extract_features(model, frames):
    with torch.no_grad():
        features = model(frames)
    return features.cpu().numpy().squeeze(0)  # 去除批次维度

# 计算 Fréchet 距离
def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    # 防止协方差矩阵奇异
    sigma1 += np.eye(sigma1.shape[0]) * 1e-6
    sigma2 += np.eye(sigma2.shape[0]) * 1e-6
    
    # 计算协方差矩阵的平方根
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        logger.warning("协方差矩阵包含非有限值，使用替代方法计算")
        covmean = np.zeros_like(covmean)
    
    # 计算Fréchet距离
    trace_val = np.trace(sigma1 + sigma2 - 2 * covmean)
    if np.iscomplexobj(trace_val):
        trace_val = trace_val.real
    
    fvd = diff.dot(diff) + trace_val
    return fvd.real if np.iscomplexobj(fvd) else fvd

# 获取文件夹中的所有视频文件
def get_video_files(folder_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
    
    if not video_files:
        logger.warning(f"文件夹 {folder_path} 中没有找到视频文件")
    
    logger.info(f"在 {folder_path} 中找到 {len(video_files)} 个视频文件")
    return video_files

# 特征统计信息
def feature_statistics(features, name):
    stats = {
        'name': name,
        'count': len(features),
        'mean': np.mean(features, axis=0).tolist(),
        'min': np.min(features).item(),
        'max': np.max(features).item(),
        'mean_magnitude': np.linalg.norm(np.mean(features, axis=0)).item(),
        'feature_dim': features.shape[1]
    }
    
    logger.info(f"{name}特征统计: {stats['count']}个样本, 特征维度: {stats['feature_dim']}")
    logger.info(f"特征值范围: [{stats['min']:.4f}, {stats['max']:.4f}], 平均幅度: {stats['mean_magnitude']:.4f}")
    
    return stats

# 主函数：计算两个文件夹视频的FVD
def calculate_fvd_from_folders(folder_path1, folder_path2, num_frames=16, max_videos=None, result_file="fvd_results.json"):
    # 初始化结果字典
    results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'folder1': folder_path1,
        'folder2': folder_path2,
        'num_frames': num_frames,
        'max_videos': max_videos,
        'fvd': None,
        'video_info': [],
        'feature_stats': {},
        'processing_time': None
    }
    
    start_time = time.time()
    
    # 初始化 I3D 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用计算设备: {device}")
    
    try:
        model = I3D().to(device)
    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        results['error'] = str(e)
        return results
    
    # 获取两个文件夹中的视频文件
    video_files1 = get_video_files(folder_path1)
    video_files2 = get_video_files(folder_path2)
    
    # 可选：限制处理的最大视频数量
    if max_videos:
        video_files1 = video_files1[:max_videos] if len(video_files1) > max_videos else video_files1
        video_files2 = video_files2[:max_videos] if len(video_files2) > max_videos else video_files2
        logger.info(f"限制处理数量: 文件夹1={len(video_files1)}, 文件夹2={len(video_files2)}")
    
    features1 = []
    features2 = []
    all_video_info = []
    
    # 处理第一个文件夹中的视频
    logger.info(f"开始处理文件夹1: {len(video_files1)}个视频")
    for i, path in enumerate(tqdm(video_files1, desc="处理文件夹1")):
        try:
            frames, video_info = extract_frames(path, num_frames)
            frames_tensor = preprocess_frames(frames).to(device)
            features = extract_features(model, frames_tensor)
            features1.append(features)
            all_video_info.append(video_info)
            
            # 每10个视频记录一次
            if (i + 1) % 10 == 0:
                logger.info(f"已处理文件夹1中 {i+1}/{len(video_files1)} 个视频")
        except Exception as e:
            logger.error(f"处理视频 {path} 时出错: {str(e)}")
            all_video_info.append({
                'path': path,
                'error': str(e)
            })
    
    # 处理第二个文件夹中的视频
    logger.info(f"开始处理文件夹2: {len(video_files2)}个视频")
    for i, path in enumerate(tqdm(video_files2, desc="处理文件夹2")):
        try:
            frames, video_info = extract_frames(path, num_frames)
            frames_tensor = preprocess_frames(frames).to(device)
            features = extract_features(model, frames_tensor)
            features2.append(features)
            all_video_info.append(video_info)
            
            # 每10个视频记录一次
            if (i + 1) % 10 == 0:
                logger.info(f"已处理文件夹2中 {i+1}/{len(video_files2)} 个视频")
        except Exception as e:
            logger.error(f"处理视频 {path} 时出错: {str(e)}")
            all_video_info.append({
                'path': path,
                'error': str(e)
            })
    
    # 检查是否有足够的特征
    if len(features1) == 0:
        error_msg = "文件夹1中没有有效的特征提取"
        logger.error(error_msg)
        results['error'] = error_msg
        return results
    
    if len(features2) == 0:
        error_msg = "文件夹2中没有有效的特征提取"
        logger.error(error_msg)
        results['error'] = error_msg
        return results
    
    logger.info(f"文件夹1提取了 {len(features1)} 个特征向量")
    logger.info(f"文件夹2提取了 {len(features2)} 个特征向量")
    
    # 合并特征
    features1 = np.array(features1)
    features2 = np.array(features2)
    
    # 记录特征统计信息
    results['feature_stats']['folder1'] = feature_statistics(features1, "文件夹1")
    results['feature_stats']['folder2'] = feature_statistics(features2, "文件夹2")
    
    # 计算均值和协方差
    logger.info("计算特征统计量...")
    mu1 = np.mean(features1, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    mu2 = np.mean(features2, axis=0)
    sigma2 = np.cov(features2, rowvar=False)
    
    # 记录协方差矩阵信息
    results['feature_stats']['folder1']['cov_shape'] = sigma1.shape
    results['feature_stats']['folder2']['cov_shape'] = sigma2.shape
    
    # 计算 FVD
    logger.info("计算Fréchet视频距离...")
    try:
        fvd = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
        results['fvd'] = fvd
        logger.info(f"计算完成: FVD = {fvd:.4f}")
    except Exception as e:
        error_msg = f"计算FVD时出错: {str(e)}"
        logger.error(error_msg)
        results['error'] = error_msg
    
    # 记录处理时间
    processing_time = time.time() - start_time
    results['processing_time'] = processing_time
    results['video_info'] = all_video_info
    
    logger.info(f"总处理时间: {processing_time:.2f}秒")
    
    # 保存结果到文件
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"结果已保存到: {result_file}")
    
    return results

# 结果解释
def interpret_fvd(fvd_value):
    if fvd_value < 1:
        return "视频内容几乎完全相同（理想情况）"
    elif fvd_value < 50:
        return "视频质量非常高，与参考视频非常接近"
    elif fvd_value < 100:
        return "视频质量良好，与参考视频相似"
    elif fvd_value < 200:
        return "视频质量一般，与参考视频有一定差异"
    elif fvd_value < 500:
        return "视频质量较差，与参考视频差异明显"
    else:
        return "视频质量很差，与参考视频完全不同"

# 示例使用
if __name__ == "__main__":
    # 替换为您的文件夹路径
    folder_path1 = '/public/home/liuhuijie/dits/Latte/test/real_test_videos'  # 真实视频文件夹
    folder_path2 = '/public/home/liuhuijie/dits/Latte/test/gen_videos'       # 生成视频文件夹
    
    # 创建结果文件名（包含时间戳）
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = f"fvd_results_{timestamp}.json"
    
    logger.info(f"开始FVD计算: {folder_path1} vs {folder_path2}")
    logger.info(f"结果将保存到: {result_file}")
    
    try:
        # 计算FVD，可选的max_videos参数用于限制处理的视频数量
        results = calculate_fvd_from_folders(
            folder_path1, 
            folder_path2, 
            num_frames=16, 
            max_videos=2048,
            result_file=result_file
        )
        
        if 'fvd' in results and results['fvd'] is not None:
            fvd_value = results['fvd']
            interpretation = interpret_fvd(fvd_value)
            print('=' * 80)
            print(f"FVD 结果: {fvd_value:.4f}")
            print(f"解释: {interpretation}")
            print('=' * 80)
            logger.info(f"最终FVD: {fvd_value:.4f} - {interpretation}")
        else:
            logger.error("未能计算出有效的FVD值")
            print("计算失败，请查看日志文件获取详细信息")
            
    except Exception as e:
        logger.exception(f"计算 FVD 时发生未处理的异常: {str(e)}")
        print(f"计算失败: {str(e)}")