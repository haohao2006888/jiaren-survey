#!/usr/bin/env python3
"""离线转写脚本 — 两轮百度 STT + 交叉比对
用法:
  python transcribe.py                    # 处理所有未转写的录音
  python transcribe.py --dry-run          # 预览模式，不实际写入
  python transcribe.py --name 张三        # 只处理指定用户
  python transcribe.py --part bio         # 只处理指定部分

依赖:
  pip install requests
  brew install ffmpeg   # macOS
"""

import base64, json, os, subprocess, sys, tempfile, time
from difflib import SequenceMatcher

try:
    import requests
except ImportError:
    print("请先安装 requests: pip install requests")
    sys.exit(1)

# ═══════════════════ 配置 ═══════════════════
BAIDU_KEY = "rrJI9DRyudEBu7NdN6JO37i1"
BAIDU_SECRET = "K53an5SVXnI7NS4yFq8hifX53d4hmqLW"
DENO_API = "https://strong-butterfly-68.haohao2006888.deno.net/"
GITHUB_RAW = "https://raw.githubusercontent.com/haohao2006888/tongxingzhe-survey/main/data/submissions.json"

# GitHub 写入配置（需要 Personal Access Token）
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = "haohao2006888/tongxingzhe-survey"
GITHUB_FILE = "data/submissions.json"

PART_MAP = {
    "bio":  {"text_field": "bio",         "audio_field": "bio_audio",   "label": "个人简介"},
    "1":    {"text_field": "part1_text",  "audio_field": "1_audio",     "label": "第一部分·原文"},
    "2":    {"text_field": "part2_text",  "audio_field": "2_audio",     "label": "第二部分·智慧"},
    "3":    {"text_field": "part3_text",  "audio_field": "3_audio",     "label": "第三部分·联想"},
}

SIMILARITY_THRESHOLD = 0.70   # 两轮结果相似度阈值
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB 上限（base64 解码后）


# ═══════════════════ 百度 Token ═══════════════════
_token_cache = {"token": "", "expires": 0}

def get_baidu_token():
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires"]:
        return _token_cache["token"]
    resp = requests.post(
        "https://aip.baidubce.com/oauth/2.0/token",
        params={"grant_type": "client_credentials",
                "client_id": BAIDU_KEY, "client_secret": BAIDU_SECRET},
        timeout=15
    )
    data = resp.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires"] = now + data.get("expires_in", 86400) - 60
    return _token_cache["token"]


# ═══════════════════ WebM → PCM ═══════════════════
def webm_to_pcm(webm_bytes):
    """将 WebM/Opus 音频转为 16kHz/16bit/mono PCM。
    返回 (pcm_bytes, error_string)"""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f_in:
        f_in.write(webm_bytes)
        webm_path = f_in.name

    pcm_path = webm_path + ".pcm"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path,
             "-f", "s16le", "-acodec", "pcm_s16le",
             "-ac", "1", "-ar", "16000", pcm_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return None, f"ffmpeg 失败: {result.stderr[:200]}"
        if not os.path.exists(pcm_path) or os.path.getsize(pcm_path) == 0:
            return None, "ffmpeg 输出为空"
        with open(pcm_path, "rb") as f:
            return f.read(), None
    except FileNotFoundError:
        return None, "ffmpeg 未安装。请运行: brew install ffmpeg"
    except subprocess.TimeoutExpired:
        return None, "ffmpeg 超时（>60s）"
    finally:
        for p in (webm_path, pcm_path):
            try: os.unlink(p)
            except OSError: pass


# ═══════════════════ 百度 STT ═══════════════════
def baidu_stt(pcm_bytes, timeout=30):
    """调用百度语音识别 REST API。
    返回 {"text": str, "err_no": int, "err_msg": str}"""
    token = get_baidu_token()
    b64 = base64.b64encode(pcm_bytes).decode()
    body = {
        "format": "pcm", "rate": 16000, "channel": 1,
        "cuid": "jiaren-transcribe", "token": token,
        "speech": b64, "len": len(pcm_bytes)
    }
    try:
        resp = requests.post(
            "https://vop.baidu.com/server_api",
            json=body, timeout=timeout
        )
        data = resp.json()
        text = "".join(data.get("result", []))
        return {"text": text, "err_no": data.get("err_no", -1),
                "err_msg": data.get("err_msg", "")}
    except Exception as e:
        return {"text": "", "err_no": -1, "err_msg": str(e)[:100]}


# ═══════════════════ 两轮比对 ═══════════════════
def transcribe_with_verification(pcm_bytes, label=""):
    """两轮独立 STT + 交叉比对，返回最优文本"""
    prefix = f"[{label}] " if label else ""

    # 第 1 轮
    print(f"  {prefix}🔊 第1轮 STT...", end=" ", flush=True)
    r1 = baidu_stt(pcm_bytes)
    t1, ok1 = r1["text"].strip(), r1["err_no"] == 0
    print(f"{'✅' if ok1 else '❌ err=' + str(r1['err_no'])} \"{t1[:60]}{'...' if len(t1)>60 else ''}\"")

    # 第 2 轮（独立请求，间隔 0.5s 防止限流）
    time.sleep(0.5)
    print(f"  {prefix}🔊 第2轮 STT...", end=" ", flush=True)
    r2 = baidu_stt(pcm_bytes)
    t2, ok2 = r2["text"].strip(), r2["err_no"] == 0
    print(f"{'✅' if ok2 else '❌ err=' + str(r2['err_no'])} \"{t2[:60]}{'...' if len(t2)>60 else ''}\"")

    # — 对比逻辑 —
    if not ok1 and not ok2:
        return {"text": "", "confidence": "low",
                "reason": f"两轮均失败: r1={r1['err_msg']}, r2={r2['err_msg']}"}
    if not ok1:
        return {"text": t2, "confidence": "medium",
                "reason": f"第1轮失败({r1['err_msg']})，采用第2轮"}
    if not ok2:
        return {"text": t1, "confidence": "medium",
                "reason": f"第2轮失败({r2['err_msg']})，采用第1轮"}
    if not t1 and not t2:
        return {"text": "", "confidence": "high",
                "reason": "两轮均未识别到语音内容"}
    if not t1:
        return {"text": t2, "confidence": "medium", "reason": "第1轮为空"}
    if not t2:
        return {"text": t1, "confidence": "medium", "reason": "第2轮为空"}

    # 两轮都有内容 → 比对
    sim = SequenceMatcher(None, t1, t2).ratio()
    print(f"  {prefix}📊 相似度: {sim:.1%}  |  len1={len(t1)} len2={len(t2)}")
    if sim >= SIMILARITY_THRESHOLD:
        # 高度一致 → 取更长/更完整的
        chosen = t1 if len(t1) >= len(t2) else t2
        return {"text": chosen, "confidence": "high",
                "reason": f"两轮一致(相似度{sim:.1%})，取较长结果"}
    else:
        # 不一致 → 标记低置信度，取更长的
        chosen = t1 if len(t1) >= len(t2) else t2
        return {"text": chosen, "confidence": "low",
                "reason": f"两轮差异大(相似度{sim:.1%})，需人工校对",
                "alt_text": t2 if chosen == t1 else t1}


# ═══════════════════ 主流程 ═══════════════════
def fetch_submissions():
    """从 Deno 代理拉取所有 submissions"""
    print("📡 拉取 submissions...")
    try:
        resp = requests.get(DENO_API, timeout=30)
        if resp.ok:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data
    except Exception as e:
        print(f"  ⚠️ Deno 代理失败: {e}")

    # 回退到 GitHub Raw
    print("  🔄 回退到 GitHub Raw...")
    try:
        resp = requests.get(GITHUB_RAW, headers={"Accept": "application/json"}, timeout=30)
        if resp.ok:
            data = resp.json()
            return data
    except Exception as e:
        print(f"  ❌ GitHub Raw 也失败: {e}")

    return None


def update_submission_via_github(submissions):
    """通过 GitHub API 写回 submissions.json"""
    if not GITHUB_TOKEN:
        print("\n⚠️  未设置 GITHUB_TOKEN 环境变量，跳过写入。")
        print("   export GITHUB_TOKEN=ghp_xxxxxxxxxxxx")
        return False

    content = json.dumps(submissions, ensure_ascii=False, indent=2)
    encoded = base64.b64encode(content.encode()).decode()

    # 先获取当前文件的 sha
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}",
               "Accept": "application/vnd.github+json"}

    resp = requests.get(url, headers=headers, timeout=30)
    if not resp.ok:
        print(f"  ❌ 获取文件信息失败: {resp.status_code} {resp.text[:200]}")
        return False

    sha = resp.json().get("sha", "")
    commit_msg = f"transcribe: {len(submissions)} submissions updated"

    body = {"message": commit_msg, "content": encoded, "sha": sha}
    put_resp = requests.put(url, headers=headers, json=body, timeout=60)
    if put_resp.ok:
        print(f"  ✅ 已写入 GitHub ({len(content)} bytes)")
        return True
    else:
        print(f"  ❌ 写入失败: {put_resp.status_code} {put_resp.text[:200]}")
        return False


def main():
    dry_run = "--dry-run" in sys.argv
    filter_name = None
    filter_part = None
    for i, a in enumerate(sys.argv):
        if a == "--name" and i + 1 < len(sys.argv):
            filter_name = sys.argv[i + 1]
        if a == "--part" and i + 1 < len(sys.argv):
            filter_part = sys.argv[i + 1]

    if dry_run:
        print("🔍 预览模式 — 不会实际写入\n")

    subs = fetch_submissions()
    if subs is None:
        print("❌ 无法获取 submissions，退出。")
        sys.exit(1)

    print(f"📋 共 {len(subs)} 条 submission\n")

    modified = False
    total_transcribed = 0

    for si, sub in enumerate(subs):
        name = sub.get("userName", "?")
        if filter_name and name != filter_name:
            continue

        st = sub.get("time", "")[:10]
        print(f"── [{si+1}/{len(subs)}] {name} ({st}) ──")

        for part_key, cfg in PART_MAP.items():
            if filter_part and part_key != filter_part:
                continue

            text_field = cfg["text_field"]
            audio_field = cfg["audio_field"]
            label = cfg["label"]

            # 跳过已有文本的
            existing = sub.get(text_field, "").strip()
            if existing:
                print(f"  ✅ {label}: 已有文字 ({len(existing)}字)，跳过")
                continue

            # 检查是否有音频
            audio_b64 = sub.get(audio_field, "")
            if not audio_b64:
                continue

            print(f"  🎙️ {label}: 发现音频，开始转写...")

            # 移除 data URL 前缀 (data:audio/webm;base64,...)
            if "," in audio_b64 and audio_b64.startswith("data:"):
                audio_b64 = audio_b64.split(",", 1)[1]

            try:
                webm_bytes = base64.b64decode(audio_b64)
            except Exception as e:
                print(f"    ❌ base64 解码失败: {e}")
                continue

            if len(webm_bytes) > MAX_AUDIO_BYTES:
                print(f"    ⚠️ 音频过大 ({len(webm_bytes)//1024}KB)，跳过")
                continue

            print(f"    📦 {len(webm_bytes)//1024}KB WebM → PCM...", end=" ", flush=True)
            pcm, err = webm_to_pcm(webm_bytes)
            if err:
                print(f"❌ {err}")
                continue
            print(f"{len(pcm)//1024}KB PCM ({len(pcm)/32000:.1f}s)")

            if dry_run:
                print(f"    🔍 [预览] 将调用百度STT转写 {part_key}")
                continue

            result = transcribe_with_verification(pcm, label=part_key)
            final_text = result["text"].strip()

            if final_text:
                sub[text_field] = final_text
                modified = True
                total_transcribed += 1
                print(f"  📝 → {text_field}: \"{final_text[:80]}{'...' if len(final_text)>80 else ''}\"")
                print(f"     置信度: {result['confidence']} | {result['reason']}")
                if result.get("alt_text"):
                    print(f"     备选: \"{result['alt_text'][:80]}{'...' if len(result['alt_text'])>80 else ''}\"")
            else:
                print(f"  ⚠️ 未识别到文字 ({result['reason']})")

    print(f"\n{'═' * 50}")
    print(f"📊 完成: 转写 {total_transcribed} 段录音")

    if dry_run:
        print("🔍 预览模式结束，未实际写入。")
        return

    if modified:
        print("💾 写回 GitHub...")
        update_submission_via_github(subs)
    else:
        print("✅ 没有需要更新的内容")


if __name__ == "__main__":
    main()
