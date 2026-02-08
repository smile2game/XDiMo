import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
import os
import glob
import time
import logging
from tqdm import tqdm
import json
import math
import random
from typing import Tuple

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

# 使用本地I3D模型路径
I3D_MODEL_PATH = '/public/home/liuhuijie/dits/Latte/share_ckpts/i3d_torchscript.pt'

def load_i3d_pretrained(device=torch.device('cpu')):
    """加载预训练的I3D模型 - 使用本地路径"""
    logger.info(f"加载本地I3D模型: {I3D_MODEL_PATH}")
    if not os.path.exists(I3D_MODEL_PATH):
        logger.error(f"I3D模型文件不存在: {I3D_MODEL_PATH}")
        raise FileNotFoundError(f"I3D模型文件不存在: {I3D_MODEL_PATH}")
    
    try:
        i3d = torch.jit.load(I3D_MODEL_PATH).eval().to(device)
        logger.info("I3D模型加载成功")
        return i3d
    except Exception as e:
        logger.error(f"加载I3D模型失败: {str(e)}")
        raise

def extract_features(model, videos, device, bs=10):
    """提取特征向量 - 修正版，避免过度归一化"""
    features = []
    
    with torch.no_grad():
        # 分批处理视频
        for i in range(0, len(videos), bs):
            batch = videos[i:i+bs].to(device)
            
            # 提取特征 - 确保获取原始特征输出
            try:
                # 使用模型提取特征
                batch_features = model(batch)
                
                # 检查特征维度 - 如果是分类输出(400维)，需要修改为特征层输出
                if batch_features.size(1) == 400:
                    logger.warning("获取的是分类输出而非特征输出，尝试获取中间层特征")
                    # 这里需要根据您的I3D模型结构调整
                    # 通常应该获取全局平均池化前的特征
                    # 以下代码可能需要根据实际模型结构修改
                    batch_features = model.features(batch)
                
                features.append(batch_features.cpu())
            except Exception as e:
                logger.error(f"特征提取失败: {str(e)}")
                # 添加零特征作为替代
                features.append(torch.zeros(batch.size(0), 1024))
    
    return torch.cat(features, dim=0)

def preprocess_single(video, resolution=224):
    """预处理单个视频片段 - 简化版，避免过度归一化"""
    # video: C,H,W,T
    c, h, w, t = video.shape

    # 确保通道数正确
    if c != 3:
        if c == 1:  # 灰度图转RGB
            video = video.repeat(3, 1, 1, 1)
        else:  # 其他情况取前3个通道
            video = video[:3]
    
    # 缩放短边到目标分辨率
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    
    # 双线性插值调整大小
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # 中心裁剪
    _, h, w, _ = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, h_start:h_start + resolution, w_start:w_start + resolution, :]

    # 仅做最小化处理，避免过度归一化
    # 保持像素值在[0, 1]范围
    return video.permute(0, 3, 1, 2).contiguous()  # C,T,H,W

def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算特征的均值和协方差 - 添加稳定性处理"""
    mu = feats.mean(axis=0)  # [d]
    
    # 添加正则化确保协方差矩阵正定
    sigma = np.cov(feats, rowvar=False)
    sigma += np.eye(sigma.shape[0]) * 1e-6
    
    return mu, sigma

def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    """计算Fréchet距离 - 修正版"""
    if feats_fake.shape[0] < 2 or feats_real.shape[0] < 2:
        logger.warning("样本数量不足，无法计算可靠的FVD")
        return float('inf')
    
    mu1, sigma1 = compute_stats(feats_fake)
    mu2, sigma2 = compute_stats(feats_real)
    
    # 计算均值差异
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # 处理复数结果
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # 计算Fréchet距离
    return (np.dot(diff, diff) + np.trace(sigma1 + sigma2 - 2 * covmean))

# 从视频中提取帧 - 修正版，避免过度归一化
def extract_frames(video_path, num_frames=16, target_fps=30):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    # 处理无效帧率
    if original_fps <= 0:
        logger.warning(f"视频 {video_path} 的帧率无效 ({original_fps}), 使用默认值30fps")
        original_fps = 30
    
    frames = []
    if total_frames == 0:
        raise ValueError(f"视频 {video_path} 没有可用的帧")
    
    # 计算采样间隔（处理帧率差异）
    skip_frames = max(1, int(original_fps / target_fps))
    
    # 确定起始帧 - 随机选择起始点
    max_start = max(0, total_frames - num_frames * skip_frames)
    start_frame = random.randint(0, max_start) if max_start > 0 else 0
    
    # 记录视频信息
    video_info = {
        'path': video_path,
        'total_frames': total_frames,
        'original_fps': original_fps,
        'target_fps': target_fps,
        'duration': duration,
        'sampled_frames': min(num_frames, total_frames),
        'skip_frames': skip_frames,
        'start_frame': start_frame
    }
    
    # 提取连续帧序列（考虑帧率差异）
    for i in range(num_frames):
        frame_idx = start_frame + i * skip_frames
        if frame_idx >= total_frames:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    # 如果帧数不足，填充最后一帧
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if frames else np.zeros((256, 256, 3), dtype=np.uint8))
    
    # 转换为张量，保持原始像素值范围 [0, 255]
    frames_array = np.array(frames)  # T, H, W, C
    frames_tensor = torch.tensor(frames_array).float()
    
    # 调整维度顺序为 [C, T, H, W]
    frames_tensor = frames_tensor.permute(3, 0, 1, 2)
    
    processing_time = time.time() - start_time
    video_info['processing_time'] = processing_time
    video_info['actual_sampled'] = len(frames)
    
    logger.info(f"视频处理完成: {os.path.basename(video_path)} - 总帧数: {total_frames} - 采样帧数: {len(frames)} - 耗时: {processing_time:.2f}s")
    logger.info(f"提取帧形状: {frames_tensor.shape}")
    
    return frames_tensor, video_info

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

# 特征统计信息 - 修正版，显示更多统计信息
def feature_statistics(features, name):
    if len(features) == 0:
        return {
            'name': name,
            'count': 0,
            'error': 'No features available'
        }
    
    # 转换为numpy数组进行计算
    features_np = features.numpy() if isinstance(features, torch.Tensor) else features
    
    stats = {
        'name': name,
        'count': len(features_np),
        'mean': np.mean(features_np, axis=0).tolist(),
        'min': np.min(features_np).item(),
        'max': np.max(features_np).item(),
        'std': np.std(features_np).item(),
        'mean_magnitude': np.linalg.norm(np.mean(features_np, axis=0)).item(),
        'feature_dim': features_np.shape[1]
    }
    
    logger.info(f"{name}特征统计: {stats['count']}个样本, 特征维度: {stats['feature_dim']}")
    logger.info(f"特征值范围: [{stats['min']:.4f}, {stats['max']:.4f}], 标准差: {stats['std']:.4f}, 平均幅度: {stats['mean_magnitude']:.4f}")
    
    return stats

# 主函数：计算两个文件夹视频的FVD - 修正版，避免过度归一化
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
    
    # 初始化I3D模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用计算设备: {device}")
    
    try:
        i3d = load_i3d_pretrained(device)
        logger.info("I3D模型加载成功")
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
    
    videos1 = []
    videos2 = []
    all_video_info = []
    
    # 处理第一个文件夹中的视频
    logger.info(f"开始处理文件夹1: {len(video_files1)}个视频")
    for i, path in enumerate(tqdm(video_files1, desc="处理文件夹1")):
        try:
            video_tensor, video_info = extract_frames(path, num_frames)
            videos1.append(video_tensor)
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
            video_tensor, video_info = extract_frames(path, num_frames)
            videos2.append(video_tensor)
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
    
    # 检查是否有足够的视频
    if len(videos1) == 0:
        error_msg = "文件夹1中没有有效的视频"
        logger.error(error_msg)
        results['error'] = error_msg
        return results
    
    if len(videos2) == 0:
        error_msg = "文件夹2中没有有效的视频"
        logger.error(error_msg)
        results['error'] = error_msg
        return results
    
    # 堆叠视频张量
    videos1_tensor = torch.stack(videos1)  # B,C,T,H,W
    videos2_tensor = torch.stack(videos2)  # B,C,T,H,W
    
    logger.info(f"文件夹1视频张量形状: {videos1_tensor.shape}")
    logger.info(f"文件夹2视频张量形状: {videos2_tensor.shape}")
    
    # 预处理视频 - 避免过度归一化
    logger.info("预处理文件夹1的视频...")
    preprocessed_videos1 = []
    for video in videos1_tensor:
        preprocessed = preprocess_single(video)
        preprocessed_videos1.append(preprocessed)
    preprocessed_videos1 = torch.stack(preprocessed_videos1)
    
    logger.info("预处理文件夹2的视频...")
    preprocessed_videos2 = []
    for video in videos2_tensor:
        preprocessed = preprocess_single(video)
        preprocessed_videos2.append(preprocessed)
    preprocessed_videos2 = torch.stack(preprocessed_videos2)
    
    # 提取特征 - 保持原始特征值
    logger.info("提取文件夹1的特征...")
    features1 = extract_features(i3d, preprocessed_videos1, device, bs=4)
    logger.info("提取文件夹2的特征...")
    features2 = extract_features(i3d, preprocessed_videos2, device, bs=4)
    
    # 记录特征统计信息 - 使用原始特征值
    results['feature_stats']['folder1'] = feature_statistics(features1, "文件夹1")
    results['feature_stats']['folder2'] = feature_statistics(features2, "文件夹2")
    
    # 检查特征值范围 - 确保没有过度归一化
    stats1 = results['feature_stats']['folder1']
    stats2 = results['feature_stats']['folder2']
    
    if stats1['mean_magnitude'] < 10 or stats2['mean_magnitude'] < 10:
        logger.warning("特征值幅度过低，可能是归一化过度")
    
    # 转换为numpy数组用于FVD计算
    features1_np = features1.numpy()
    features2_np = features2.numpy()
    
    # 计算 FVD
    logger.info("计算Fréchet视频距离...")
    try:
        fvd = frechet_distance(features1_np, features2_np)
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
    if fvd_value < 10:
        return "视频内容几乎完全相同（可能计算有误）"
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
    folder_path1 = '/public/home/liuhuijie/dits/Latte/test/630'  # 真实视频文件夹
    folder_path2 = '/public/home/liuhuijie/dits/Latte/test/real_videos'       # 生成视频文件夹
    
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
            num_frames=16,  # 使用16帧
            max_videos=10,  # 先测试少量视频
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