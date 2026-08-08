#!/usr/bin/env python3
"""
MeloMind Web - 网易云音乐驱动版
启动前需确保本地已部署 NeteaseCloudMusicApi 服务（默认地址 http://localhost:3000）
"""
import os
import sys
import json
import tempfile
import numpy as np
import librosa
from fastdtw import fastdtw
from flask import Flask, render_template, request, jsonify, send_file, url_for
import requests
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ---------- 网易云 API 基础配置 ----------
NCM_API = "http://localhost:3000"   # NeteaseCloudMusicApi 服务地址

def ncm_get(path, params=None):
    """简单封装，调用网易云 API"""
    try:
        resp = requests.get(f"{NCM_API}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"API error: {e}")
        return None

# ---------- 风格共鸣：直接使用网易云相似歌曲 ----------
def search_ncm_song(keywords):
    """搜索歌曲，返回第一个结果的 ID"""
    data = ncm_get("/search", {"keywords": keywords, "type": 1, "limit": 1})
    if data and data.get("code") == 200 and data.get("result", {}).get("songs"):
        song = data["result"]["songs"][0]
        return {
            "id": song["id"],
            "name": song["name"],
            "artists": ", ".join(ar["name"] for ar in song["ar"]),
            "cover": song["al"]["picUrl"]
        }
    return None

def get_similar_songs(song_id, limit=10):
    """获取相似歌曲列表"""
    data = ncm_get("/simi/song", {"id": song_id})
    if data and data.get("code") == 200:
        songs = data.get("songs", [])
        results = []
        for s in songs[:limit]:
            results.append({
                "name": s["name"],
                "id": s["id"],
                "artists": ", ".join(ar["name"] for ar in s["artists"]),
                "cover": s["album"]["picUrl"],
                "audio_url": f"https://music.163.com/song/media/outer/url?id={s['id']}.mp3"  # 官方试听链接
            })
        return results
    return []

# ---------- 旋律追踪：本地 DTW + 网易云信息（简化版，索引需由用户上传建立） ----------
# 我们可以维护一个全局的旋律索引，但数据源来自用户上传的歌曲（可附带网易云 ID）
# 此处保留你原有的 DTW 逻辑，但将库文件信息与网易云歌曲关联。

MELODY_INDEX = {
    "features": [],
    "paths": [],        # 本地临时文件路径
    "ncm_ids": [],      # 对应的网易云歌曲 ID（如果有）
    "ncm_info": []      # 歌曲名/歌手等
}

# 为了演示，我们提供一个通过网易云 ID 缓存试听片段并建索引的函数
def add_to_melody_index(ncm_song_id):
    """通过网易云歌曲 ID 获取试听音频，提取旋律轮廓，加入索引"""
    url = f"https://music.163.com/song/media/outer/url?id={ncm_song_id}.mp3"
    try:
        # 下载试听片段（通常几分钟，我们只取前 30 秒）
        import io
        import soundfile as sf
        resp = requests.get(url, timeout=15, stream=True)
        if resp.status_code != 200:
            return False
        # 将音频流加载为 numpy
        y, sr = librosa.load(io.BytesIO(resp.content), sr=22050, mono=True, duration=30)
        # 提取基频
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                fmax=librosa.note_to_hz('C7'), sr=sr)
        f0 = np.nan_to_num(f0, nan=0.0)
        midi = np.where(f0 > 0, 69 + 12 * np.log2(f0 / 440.0), 0).astype(np.float32)
        if np.count_nonzero(midi) < 10:
            return False
        MELODY_INDEX["features"].append(midi)
        MELODY_INDEX["ncm_ids"].append(ncm_song_id)
        # 保存一份音频文件以便播放（可选）
        # 这里省略
        return True
    except Exception as e:
        print(f"Failed to index {ncm_song_id}: {e}")
        return False

# 初始化时为一些热门歌曲建索引（演示用）
def preload_hot_songs():
    hot_ids = [41630490, 35476049, 28306219, 108253]  # 示例 ID
    for sid in hot_ids[:3]:
        add_to_melody_index(sid)

# 旋律搜索函数（复用之前的 DTW，但结果返回网易云信息）
def melody_search(audio_path, top_n=5):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'), sr=sr)
    f0 = np.nan_to_num(f0, nan=0.0)
    q_pitch = np.where(f0 > 0, 69 + 12 * np.log2(f0 / 440.0), 0).astype(np.float32)
    q_sub = q_pitch[::10]
    distances = []
    for feat in MELODY_INDEX["features"]:
        f_sub = feat[::10]
        dist, _ = fastdtw(q_sub.reshape(-1,1), f_sub.reshape(-1,1), radius=30)
        distances.append(dist)
    if not distances:
        return []
    top_idx = np.argsort(distances)[:top_n]
    results = []
    for idx in top_idx:
        nid = MELODY_INDEX["ncm_ids"][idx]
        # 获取歌曲详细信息
        song_detail = ncm_get("/song/detail", {"ids": str(nid)})
        song = None
        if song_detail and song_detail["code"] == 200 and song_detail["songs"]:
            song = song_detail["songs"][0]
        if song:
            results.append({
                "name": song["name"],
                "artists": ", ".join(ar["name"] for ar in song["ar"]),
                "cover": song["al"]["picUrl"],
                "audio_url": f"https://music.163.com/song/media/outer/url?id={nid}.mp3",
                "score": distances[idx]
            })
    return results

# ---------- 路由 ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    mode = request.form.get('mode', 'style')
    # 风格模式：接受文本搜索或已上传音频
    if mode == 'style':
        # 优先检查是否上传了音频文件（可以调用识别 API 得到歌名，但这里简化为文本）
        keywords = request.form.get('keywords', '').strip()
        if 'file' in request.files and request.files['file'].filename != '':
            # 如果上传了文件，我们暂时用文件名作为关键词去搜索
            file = request.files['file']
            # 实际上可以调用听歌识曲 API，这里简单用文件名
            keywords = os.path.splitext(file.filename)[0]
        if not keywords:
            return jsonify({"error": "请提供歌曲名称或上传音频文件"}), 400
        # 搜索网易云歌曲
        song_info = search_ncm_song(keywords)
        if not song_info:
            return jsonify({"error": "没有找到匹配的歌曲"}), 404
        # 获取相似推荐
        similar = get_similar_songs(song_info["id"])
        return jsonify({
            "mode": "style",
            "query": song_info,
            "results": similar
        })

    elif mode == 'melody':
        if 'file' not in request.files:
            return jsonify({"error": "请上传哼唱片段或音频"}), 400
        file = request.files['file']
        _, ext = os.path.splitext(file.filename)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        try:
            results = melody_search(tmp_path, top_n=10)
            return jsonify({"mode": "melody", "results": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.unlink(tmp_path)
    else:
        return jsonify({"error": "Invalid mode"}), 400

@app.route('/audio/<path:relpath>')
def serve_audio(relpath):
    # 仅用于本地文件，现在大部分试听直接跳转网易云链接，此路由可废弃
    return "use remote url", 404

if __name__ == '__main__':
    # 启动时预加载一些索引（可选）
    # preload_hot_songs()
    app.run(debug=True, host='0.0.0.0', port=5000)
