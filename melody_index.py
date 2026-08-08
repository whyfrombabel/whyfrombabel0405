import numpy as np
import librosa
from fastdtw import fastdtw
import os
import json

# 预加载的旋律索引（实际应用中可保存到磁盘）
_MELODY_DB = []   # 列表元素: {'path': ..., 'contour': np.array, 'song_info': {...}}

def extract_pitch_contour(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < sr * 0.5:
        raise ValueError("音频太短")
    f0, _, _ = librosa.pyin(y,
                            fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'),
                            sr=sr)
    f0 = np.nan_to_num(f0, nan=0.0)
    midi = np.where(f0 > 0, 69 + 12 * np.log2(f0 / 440.0), 0)
    return midi.astype(np.float32)

def build_index_from_ncm(hot_song_ids):
    """从网易云热门歌曲构建旋律索引（示例，实际需异步实现）"""
    # 这里简化：直接存储一个假的索引文件用于演示
    global _MELODY_DB
    _MELODY_DB = []  # 实际应当下载音频片段并提取
    # 略...

def search_melody(query_audio, top_n=10, genre=None):
    # 提取查询旋律
    q_pitch = extract_pitch_contour(query_audio)
    q_sub = q_pitch[::10]
    distances = []
    for item in _MELODY_DB:
        db_pitch = item['contour']
        db_sub = db_pitch[::10]
        dist, _ = fastdtw(q_sub.reshape(-1,1), db_sub.reshape(-1,1), radius=30)
        distances.append((dist, item))
    distances.sort(key=lambda x: x[0])
    top = distances[:top_n]
    results = []
    for dist, item in top:
        results.append({
            'name': item['song_info']['name'],
            'artist': item['song_info']['artist'],
            'cover': item['song_info']['cover'],
            'play_url': item['song_info']['url'],
            'distance': dist
        })
    return results
