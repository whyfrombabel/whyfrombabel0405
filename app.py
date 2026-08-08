#!/usr/bin/env python3
"""
MeloMind Web - 双核音乐发现引擎网页版
启动前请设置环境变量 MUSIC_LIBRARY_PATH 指向你的音乐库目录，
首次启动会自动建立风格和旋律索引（可能需要数分钟）。
"""
import os
import sys
import tempfile
import numpy as np
import librosa
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from flask import Flask, render_template, request, jsonify, send_file, url_for
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ---------- 音频特征提取 ----------
def extract_style_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < sr * 0.5:
        raise ValueError("Audio too short")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean, cent_std = np.mean(cent), np.std(cent)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean, zcr_std = np.mean(zcr), np.std(zcr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean, rolloff_std = np.mean(rolloff), np.std(rolloff)
    return np.concatenate([mfcc_mean, mfcc_std,
                           [cent_mean, cent_std, zcr_mean, zcr_std, rolloff_mean, rolloff_std]])

def extract_pitch_contour(file_path, sr=22050, hop_length=512):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < sr * 0.5:
        raise ValueError("Audio too short")
    f0, _, _ = librosa.pyin(y,
                            fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'),
                            sr=sr, hop_length=hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)
    midi = np.where(f0 > 0, 69 + 12 * np.log2(f0 / 440.0), 0)
    return midi.astype(np.float32)

# ---------- 全局索引管理 ----------
STYLE_INDEX = None
MELODY_INDEX = None

def build_or_load_indices():
    global STYLE_INDEX, MELODY_INDEX
    lib_dir = os.environ.get("MUSIC_LIBRARY_PATH", "./music_library")
    if not os.path.isdir(lib_dir):
        print(f"WARNING: 音乐库目录 {lib_dir} 不存在，请设置 MUSIC_LIBRARY_PATH。")
        return

    # 风格索引
    style_cache = os.path.join(lib_dir, "style_index.npz")
    if os.path.exists(style_cache):
        data = np.load(style_cache, allow_pickle=True)
        STYLE_INDEX = {
            "features": data['features'],
            "paths": data['paths'],
            "scaler": StandardScaler()
        }
        STYLE_INDEX['scaler'].mean_ = data['scaler_mean']
        STYLE_INDEX['scaler'].scale_ = data['scaler_scale']
        print(f"Loaded style index: {len(data['paths'])} tracks")
    else:
        print("Building style index...")
        features, paths = [], []
        for f in _iter_audio(lib_dir):
            try:
                feat = extract_style_features(f)
                features.append(feat)
                paths.append(f)
            except Exception:
                pass
        if features:
            features = np.array(features)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            np.savez(style_cache, features=features_scaled, paths=np.array(paths),
                     scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
            STYLE_INDEX = {"features": features_scaled, "paths": np.array(paths), "scaler": scaler}
            print(f"Style index built: {len(paths)} tracks")
        else:
            print("No audio files found for style index.")

    # 旋律索引
    melody_cache = os.path.join(lib_dir, "melody_index.npz")
    if os.path.exists(melody_cache):
        data = np.load(melody_cache, allow_pickle=True)
        MELODY_INDEX = {"features": data['features'], "paths": data['paths']}
        print(f"Loaded melody index: {len(data['paths'])} tracks")
    else:
        print("Building melody index...")
        contours, paths = [], []
        for f in _iter_audio(lib_dir):
            try:
                pitch = extract_pitch_contour(f)
                if np.count_nonzero(pitch) < 10:
                    continue
                contours.append(pitch)
                paths.append(f)
            except Exception:
                pass
        if contours:
            np.savez(melody_cache, features=np.array(contours, dtype=object), paths=np.array(paths))
            MELODY_INDEX = {"features": contours, "paths": np.array(paths)}
            print(f"Melody index built: {len(paths)} tracks")
        else:
            print("No audio files found for melody index.")

def _iter_audio(root_dir):
    valid_ext = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(valid_ext):
                yield os.path.join(dirpath, fn)

# ---------- 路由 ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    mode = request.form.get('mode', 'style')  # 'style' 或 'melody'
    if mode not in ('style', 'melody'):
        return jsonify({"error": "Invalid mode"}), 400

    # 保存临时文件
    _, ext = os.path.splitext(file.filename)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        if mode == 'style' and STYLE_INDEX is not None:
            feat = extract_style_features(tmp_path)
            q_scaled = STYLE_INDEX['scaler'].transform([feat])[0]
            sims = [1 - cosine(q_scaled, f) for f in STYLE_INDEX['features']]
            top_n = min(int(request.form.get('top', 5)), 20)
            top_idx = np.argsort(sims)[::-1][:top_n]
            results = []
            for idx in top_idx:
                path = STYLE_INDEX['paths'][idx]
                results.append({
                    "name": os.path.basename(path),
                    "score": float(sims[idx]),
                    "audio_url": url_for('serve_audio', relpath=os.path.relpath(path, start=MUSIC_LIBRARY))
                })
            return jsonify({"mode": "style", "results": results})

        elif mode == 'melody' and MELODY_INDEX is not None:
            q_pitch = extract_pitch_contour(tmp_path)
            q_sub = q_pitch[::10]
            distances = []
            for feat in MELODY_INDEX['features']:
                f_sub = feat[::10]
                dist, _ = fastdtw(q_sub.reshape(-1,1), f_sub.reshape(-1,1), radius=30)
                distances.append(dist)
            top_n = min(int(request.form.get('top', 5)), 20)
            top_idx = np.argsort(distances)[:top_n]
            results = []
            for idx in top_idx:
                path = MELODY_INDEX['paths'][idx]
                results.append({
                    "name": os.path.basename(path),
                    "score": float(distances[idx]),
                    "audio_url": url_for('serve_audio', relpath=os.path.relpath(path, start=MUSIC_LIBRARY))
                })
            return jsonify({"mode": "melody", "results": results})
        else:
            return jsonify({"error": "索引未就绪，请检查音乐库配置"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)  # 清理临时文件

@app.route('/audio/<path:relpath>')
def serve_audio(relpath):
    # 安全检查：限制只能访问音乐库内的文件
    lib_dir = os.path.abspath(MUSIC_LIBRARY)
    requested = os.path.abspath(os.path.join(lib_dir, relpath))
    if not requested.startswith(lib_dir):
        return "Forbidden", 403
    if not os.path.isfile(requested):
        return "File not found", 404
    return send_file(requested, mimetype='audio/*')

# ---------- 启动 ----------
if __name__ == '__main__':
    MUSIC_LIBRARY = os.environ.get("MUSIC_LIBRARY_PATH", "./music_library")
    os.makedirs(MUSIC_LIBRARY, exist_ok=True)
    build_or_load_indices()
    app.run(debug=True, host='0.0.0.0', port=5000)
