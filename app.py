import os
import tempfile
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from database import init_db, get_db, query_history, save_history
from melody_index import search_melody
from style_engine import recommend_style
import requests
import uuid

app = Flask(__name__)
app.secret_key = 'change-this-to-a-random-secret'

# ---------- 网易云 API 基础地址 ----------
NCM_API = 'http://localhost:3000'

def ncm_get(endpoint, params=None):
    try:
        resp = requests.get(f'{NCM_API}{endpoint}', params=params, timeout=10)
        return resp.json()
    except:
        return None

# ---------- 用户系统 ----------
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400
    db = get_db()
    try:
        db.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                   (username, generate_password_hash(password)))
        db.commit()
        return jsonify({'msg': '注册成功'})
    except:
        return jsonify({'error': '用户名已存在'}), 409

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    db = get_db()
    user = db.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,)).fetchone()
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': '用户名或密码错误'}), 401
    session['user_id'] = user['id']
    session['username'] = username
    return jsonify({'msg': '登录成功', 'username': username})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'msg': '已登出'})

@app.route('/api/me')
def me():
    if 'user_id' in session:
        return jsonify({'logged_in': True, 'username': session['username']})
    return jsonify({'logged_in': False})

@app.route('/api/history')
def history():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    rows = query_history(session['user_id'])
    return jsonify([dict(r) for r in rows])

# ---------- 搜索核心 ----------
@app.route('/api/search/melody', methods=['POST'])
def melody_search():
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400
    file = request.files['file']
    genre = request.form.get('genre', '')  # 可选流派
    # 保存临时文件
    _, ext = os.path.splitext(file.filename)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)
    try:
        results = search_melody(tmp_path, top_n=10, genre=genre)
        # 保存搜索历史
        if 'user_id' in session:
            save_history(session['user_id'], 'melody', file.filename, str(results))
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)

@app.route('/api/search/style', methods=['POST'])
def style_search():
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400
    file = request.files['file']
    genre = request.form.get('genre', '')
    _, ext = os.path.splitext(file.filename)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)
    try:
        results = recommend_style(tmp_path, top_n=10, genre=genre)
        if 'user_id' in session:
            save_history(session['user_id'], 'style', file.filename, str(results))
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)

# ---------- 额外工具：从网易云搜索一首歌，获取播放链接 ----------
@app.route('/api/song/play')
def get_play_url():
    q = request.args.get('q')
    if not q:
        return jsonify({'error': '缺少参数 q'}), 400
    res = ncm_get('/search', {'keywords': q, 'type': 1, 'limit': 1})
    if not res or res.get('code') != 200:
        return jsonify({'error': '搜索失败'}), 500
    songs = res.get('result', {}).get('songs')
    if not songs:
        return jsonify({'error': '未找到歌曲'}), 404
    song = songs[0]
    # 获取播放URL
    url_res = ncm_get('/song/url', {'id': song['id']})
    play_url = ''
    if url_res and url_res.get('code') == 200 and url_res.get('data'):
        play_url = url_res['data'][0].get('url', '')
    return jsonify({
        'id': song['id'],
        'name': song['name'],
        'artist': ', '.join(ar['name'] for ar in song['ar']),
        'cover': song['al']['picUrl'],
        'url': play_url
    })

# ---------- 页面路由 ----------
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
