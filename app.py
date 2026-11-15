# -*- coding: utf-8 -*-
"""手机端 TTS 测试版：浏览器打开 http://127.0.0.1:8080 ，输入文字即可流式朗读。"""
import os, sys, json, time, struct, threading, subprocess, select
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault('is_g2pw', '0')

import tts_mobile

ENG = None
DEC = None
DEC_LOCK = threading.Lock()
# 整个合成流程必须串行化。
# ThreadingHTTPServer 会给每个请求开线程，客户端一次回复十几句就连着发十几个请求，
# 于是十几个合成线程一起抢 GIL 和 CPU，连 accept 都轮不上——表现就是
# 后面所有连接 connect timeout、整段没声音，服务看起来"死了"但进程还在。
SYNTH_LOCK = threading.Lock()
SR = 32000

def start_dec():
    global DEC
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = os.path.join(_HERE, 'bin')
    tmpz = os.path.join(_HERE, 'io', 'dummy_z.bin')
    tmpg = os.path.join(_HERE, 'io', 'dummy_ge.bin')
    np.zeros(192*8, dtype=np.float32).tofile(tmpz)
    ENG.ge_.astype(np.float32).tofile(tmpg)
    DEC = subprocess.Popen(
        [os.path.join(_HERE,'bin','dec_server'), os.path.join(_HERE,'segs')+'/',
         tmpz, tmpg, 'x', '64', '--chunk', '64', '--ctx', '16', '--serve'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=open(os.path.join(_HERE, 'io', 'dec_err.log'), 'wb'), env=env)
    print('[app] dec_server started pid=%d' % DEC.pid, flush=True)
    # 探活：segs 缺段时它会立刻退出，而错误只写进 io/dec_err.log（每次启动被覆盖）。
    # 不看一眼的话，用户要到第一次合成才看到 BrokenPipeError。
    time.sleep(0.6)
    if DEC.poll() is not None:
        import sys as _sys
        print('[app] !! dec_server 启动失败（exit=%s），错误如下：' % DEC.returncode, flush=True)
        try:
            with open(os.path.join(_HERE, 'io', 'dec_err.log'), errors='replace') as _ef:
                print(_ef.read()[-3000:], flush=True)
        except Exception:
            pass
        _sys.exit(1)

def _readn(fd, n, timeout=120.0):
    # 带超时读满 n 字节。
    # 原来是不带超时的阻塞读：dec_server 一旦卡死，这里永久阻塞，而
    # SYNTH_LOCK 还被持有 —— 后面所有请求全部排队等死，服务看着活着其实已经废了。
    buf = b""
    deadline = time.time() + timeout
    while len(buf) < n:
        remain = deadline - time.time()
        if remain <= 0:
            raise IOError("dec_server read timeout")
        try:
            r, _, _ = select.select([fd], [], [], min(remain, 1.0))
        except Exception:
            r = [fd]
        if not r:
            continue
        c = os.read(fd, n - len(buf))
        if not c:
            raise IOError("pipe closed")
        buf += c
    return buf


def _restart_dec():
    # dec 的协议是严格时序的，一旦错位后面每句都会错，必须整个重开。
    global DEC
    try:
        if DEC and DEC.poll() is None:
            DEC.kill()
    except Exception:
        pass
    try:
        start_dec()
    except Exception as e:
        print("[dec] restart failed: %s" % e, flush=True)

def dec_stream(z):
    """把 z 交给常驻 dec，逐块 yield float32 音频。

    协议是严格时序的：一旦中途出错（超时、管道断、客户端提前断开），
    dec_server 的 stdin 里会残留本次 payload，后面每句都会错位。
    所以异常一律重启 dec 进程，而不是继续用。
    """
    fd = DEC.stdout.fileno()
    T = z.shape[2]
    payload = np.ascontiguousarray(z, dtype=np.float32).tobytes()
    try:
        DEC.stdin.write(struct.pack('<i', T))
        DEC.stdin.write(payload)
        DEC.stdin.flush()
        while True:
            hdr = _readn(fd, 4)
            n = struct.unpack('<i', hdr)[0]
            if n < 0:
                break
            raw = _readn(fd, n * 4)
            yield np.frombuffer(raw, dtype=np.float32)
    except (IOError, OSError, ValueError, struct.error) as e:
        # 只有这里能拿到 DEC_LOCK，重启是安全的
        print('[dec] !! %s —— 重启 dec_server' % e, flush=True)
        _restart_dec()
        raise


# ============ 段级流式：切分 / 静音裁剪 / 标点补静音 ============
MAX_SEG_CHARS = 55
FIRST_SEG_CHARS = 12          # 第一段更短 -> 首包更快（8 字太碎，常把词切断）
MIN_TAIL_CHARS = 6            # 切完剩下的尾巴太短就不切了（三五个字的碎段最容易让 s1 复读）
SENT_END = "。！？；"
SOFT_CUT = "，、："
# 段尾停顿。原来 0.26/0.14/0.08 在「一段一句」时正好，
# 但换行多了以后每段后面都垫一下，听感就变成一顿一顿的。
# 压到原来的一半左右，靠标点本身的气口就够了。
TRAIL_GAPS = ((("。", "！", "？"), 0.14), (("，", "、", "；", "："), 0.07), (("…", "—"), 0.18))
DEFAULT_GAP = 0.04
SIL_THRESH = 0.004
SIL_FRAME = 0.02
LEAD_KEEP = 0.02
TAIL_HOLD = 0.04

def _split_sentences(text):
    out, buf = [], ""
    for ch in text:
        buf += ch
        if ch in SENT_END:
            out.append(buf); buf = ""
    if buf: out.append(buf)
    return out

def _hard_cut(s, max_len):
    lo = max(1, max_len - 6)
    for k in range(max_len, lo - 1, -1):
        if s[k-1] in SOFT_CUT:
            return s[:k], s[k:]
    # 找不到标点就不切了 —— 从句子中间劈开会让她的话听着一顿一顿的
    return "", s



# 一段别超过这么长。
# 实测：codes 超过 ~120 之后（z 帧数 = 2×codes，dec 按 48 帧一块推进），
# 从第 6 块开始输出就变成又平又直的嗡嗡（包络 cv 从 0.9 掉到 0.13）。
# 20 个字 ≈ 40 音素 ≈ 90 codes ≈ 3.5 秒，稳稳在安全区里。
LONG_SENT = 22


def _cut_on_comma(s, max_len):
    """只在标点上切。找不到标点就整句留着，绝不从句子中间劈开。

    用户反馈：「按标点符号分，别在句子中间断开，那样听感极差」。
    所以这里没有任何按字数的硬切 —— 宁可合成慢半秒，也不能把词劈成两半。
    """
    out, buf = [], ""
    for ch in s:
        buf += ch
        if ch in SENT_END or ch in SOFT_CUT:
            if len(buf) >= max_len * 0.6:
                out.append(buf)
                buf = ""
    if buf:
        if out and len(buf) < 6:
            out[-1] += buf          # 剩个三五个字的碎尾巴并回上一段
        else:
            out.append(buf)
    return out


def plan_segments(text):
    """按标点分句，一句一段。

    短句整句合成；只有超过 LONG_SENT 的长句才在逗号处再分，
    而且分点一定是标点。没有标点就一直整句合成下去。
    """
    t = (text or "").replace("\r", "").replace("\n", "").strip()
    if not t:
        return []
    segs = []
    for sent in _split_sentences(t):
        if len(sent) <= LONG_SENT:
            segs.append(sent)
        else:
            segs.extend(_cut_on_comma(sent, LONG_SENT))
    return [s for s in segs if s.strip()]

def audio_metrics(x):
    """过零率 + 6kHz 以上能量占比。用来抓电子震动音。

    实测正常语音：zcr≈0.14、hf6k≈0.25
    谱平坦度对这段 TTS 输出没区分度（清音帧本来就接近 1），所以不用它。
    """
    s = np.sign(x)
    s[s == 0] = 1
    zcr = float((np.abs(np.diff(s)) > 0).mean())
    X = np.abs(np.fft.rfft(x))
    fr = np.fft.rfftfreq(len(x), 1.0 / SR)
    hf = float(X[fr > 6000].sum() / (X.sum() + 1e-9))
    return zcr, hf


def envelope_cv(x, frame=320):
    """包络变异系数 —— 专门抓"又平又长、没有快慢"的嗡鸣。

    人说话的音节有起伏，10ms 帧的 RMS 序列 CV 通常 0.5~1.2；
    嗡鸣是一段稳定的音调，CV 会掉到 0.25 以下。
    这个是 zcr/hf6k 抓不到的（嗡鸣是低频，过零率不高、6k 以上能量也不多）。
    """
    n = len(x) // frame
    if n < 4:
        return 0.0
    rms = np.sqrt(np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1))
    m = float(rms.mean())
    if m < 1e-6:
        return 0.0
    return float(rms.std() / m)


def min_window_cv(x, win=0.6, frame=320):
    """整段平均 CV 会被正常语音盖住，这里用滑动窗口找"最平"的那 0.6 秒。

    一句话里只有句尾两个字变嗡鸣时，整段 cv 还有 0.9，看不出问题；
    但那个窗口的 cv 会掉到 0.1 左右。这才是能定位问题的判据。
    """
    n = len(x) // frame
    if n < 6:
        return 0.0
    rms = np.sqrt(np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1))
    w = max(6, int(win * SR / frame))
    if n < w:
        m = float(rms.mean())
        return 0.0 if m < 1e-6 else float(rms.std() / m)
    best = 9.9
    step = max(1, w // 3)
    for i in range(0, n - w + 1, step):
        seg = rms[i:i + w]
        m = float(seg.mean())
        if m < 1e-5:
            continue
        cv = float(seg.std() / m)
        if cv < best:
            best = cv
    return 0.0 if best > 9 else best





def trailing_gap(seg):
    t = seg.rstrip()
    if t:
        c = t[-1]
        for chars, gap in TRAIL_GAPS:
            if c in chars: return gap
    return DEFAULT_GAP

HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TTS 测试版</title>
<style>
*{box-sizing:border-box}
body{margin:0;padding:16px;font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;
     background:#111;color:#eee;min-height:100vh}
h1{font-size:17px;margin:0 0 12px;font-weight:600;color:#8ab4f8}
textarea{width:100%;height:120px;padding:12px;font-size:16px;border-radius:10px;
         border:1px solid #333;background:#1a1a1a;color:#eee;resize:vertical;outline:none}
textarea:focus{border-color:#8ab4f8}
.row{display:flex;gap:10px;margin-top:12px}
button{flex:1;padding:14px;font-size:16px;font-weight:600;border:0;border-radius:10px;
       background:#8ab4f8;color:#111;cursor:pointer}
button:disabled{background:#333;color:#666;cursor:not-allowed}
button.stop{background:#c0392b;color:#fff}
#log{margin-top:14px;font-size:13px;color:#888;line-height:1.7;white-space:pre-wrap;
     font-family:ui-monospace,monospace}
.bar{height:4px;background:#222;border-radius:2px;overflow:hidden;margin-top:12px}
.bar i{display:block;height:100%;width:0;background:#8ab4f8;transition:width .2s}
</style></head><body>
<h1>🎤 TTS 测试版</h1>
<textarea id="txt" placeholder="输入要朗读的文字...">今天天气很好，我终于把这个东西跑通了。</textarea>
<div class="row">
  <button id="go">朗读</button>
  <button id="stop" class="stop">停止</button>
</div>
<div class="bar"><i id="prog"></i></div>
<div id="log">就绪</div>
<script>
const SR = 32000;
let ctx = null, sources = [], playing = false, nextTime = 0, abortCtl = null;

function log(m){ document.getElementById('log').textContent = m; }

function ensureCtx(){
  if (!ctx || ctx.state === 'closed') { ctx = new (window.AudioContext||window.webkitAudioContext)({sampleRate: SR}); nextTime = 0; }
  if (ctx.state === 'suspended') return ctx.resume();
  return Promise.resolve();
}

function playChunk(f32){
  if (!f32 || f32.length === 0) return;
  const buf = ctx.createBuffer(1, f32.length, SR);
  buf.copyToChannel(f32, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const now = ctx.currentTime;
  if (nextTime < now + 0.05) nextTime = now + 0.05;
  src.start(nextTime);
  nextTime += buf.duration;
  sources.push(src);
}

async function speak(){
  const text = document.getElementById('txt').value.trim();
  if (!text) return;
  await ensureCtx();
  document.getElementById('go').disabled = true;
  playing = true;
  nextTime = 0;
  abortCtl = new AbortController();
  const t0 = performance.now();
  let got = 0, firstAt = 0;
  try {
    const resp = await fetch('/tts', {method:'POST', body: text,
      signal: abortCtl.signal, headers:{'Content-Type':'text/plain;charset=utf-8'}});
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const reader = resp.body.getReader();
    let carry = new Uint8Array(0);
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      let merged;
      if (carry.length) {
        merged = new Uint8Array(carry.length + value.length);
        merged.set(carry); merged.set(value, carry.length);
      } else merged = value;
      const usable = merged.length - (merged.length % 2);
      carry = merged.slice(usable);
      if (usable === 0) continue;
      const vi = new Int16Array(merged.buffer, merged.byteOffset, usable/2);
      const f32 = new Float32Array(vi.length);
      for (let i=0;i<vi.length;i++) f32[i] = vi[i] / 32768;
      if (!firstAt) firstAt = performance.now() - t0;
      got += f32.length;
      playChunk(f32);
      log('首包 ' + Math.round(firstAt) + ' ms | 已收 ' + (got/SR).toFixed(2) + ' s');
    }
    log('完成：' + (got/SR).toFixed(2) + ' s 音频，首包 ' + Math.round(firstAt) + ' ms，总 ' + Math.round(performance.now()-t0) + ' ms');
  } catch(e) {
    if (e.name !== 'AbortError') log('错误: ' + e.message);
    else log('已停止');
  } finally {
    document.getElementById('go').disabled = false;
    playing = false;
  }
}

function stopAll(){
  if (abortCtl) { try{abortCtl.abort();}catch(e){} }
  for (const s of sources) { try{ s.stop(); }catch(e){} }
  sources = []; nextTime = 0;
  log('已停止');
}

document.getElementById('go').onclick = speak;
document.getElementById('stop').onclick = stopAll;
</script></body></html>"""

LAST_REQ = [0.0]     # 最后一次合成请求的时间，给保活线程看


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            b = HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_response(404); self.send_header('Content-Length','0'); self.end_headers()

    def do_POST(self):
        LAST_REQ[0] = time.time()
        n = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(n).decode('utf-8', 'replace').strip()
        # 支持两种 body：纯文本（老接口） / JSON（带情绪参考与语速）
        ref_id = None
        speed = 1.0
        aux = 0.0
        gap = 1.0
        text = raw
        if raw.startswith('{'):
            try:
                j = json.loads(raw)
                text = (j.get('text') or '').strip()
                ref_id = j.get('ref') or None
                # 边界必须夹住：s=0.001 会让码表放大 1000 倍，
                # a=1e6 在 mix_ge 里做列表乘法直接 MemoryError，
                # g=1e3 会 np.zeros 出几个 G 的静音。
                speed = min(2.0, max(0.5, float(j.get('s') or 1.0)))
                aux = min(7.0, max(0.0, float(j.get('a') or 0.0)))
                gap = min(3.0, max(0.0, float(j.get('g') or 1.0)))
            except Exception:
                # JSON 坏了就明确报错。原来静默 pass，会把整段 JSON 当正文合成出来。
                self.send_response(400)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
        if not text:
            self.send_response(400); self.send_header('Content-Length','0'); self.end_headers(); return
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Transfer-Encoding', 'chunked')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        t0 = time.time()
        first = None
        total = 0
        # 注意：这里不能逐块做线性插值重采样。
        # emit 是按 4096 样本的块调用的，逐块 linspace 会让块边界相位跳变 -> 爆音，
        # 线性插值降采样还会混叠。语速先交给播放端（AudioTrack 可变速不变调）。
        seg_buf = []          # 当前段的样本缓冲（为了整段变速）

        def _fade(x, ms=20.0):
            """段首尾各加几毫秒淡入淡出。段是硬拼的，不加会在边界跳变出咔哒声。"""
            n = int(SR * ms / 1000.0)
            if n <= 1 or len(x) < n * 2:
                return x
            y = x.copy()
            w = np.linspace(0.0, 1.0, n, dtype=np.float32)
            y[:n] *= w
            y[-n:] *= w[::-1]
            return y

        # —— 录音留证（调试用，默认关）——
        # 开启后每次生成的音频和文本都落盘，排查音质问题用。
        # 注意：会把用户输入的原文写进 tts_dumps/*.txt，默认关闭。
        # 需要时设 KATTS_DUMP_AUDIO=1。
        dump_path = None
        try:
            if os.environ.get('KATTS_DUMP_AUDIO', '0') != '1':
                raise RuntimeError('dump disabled')
            _dd = os.path.join(_HERE, 'tts_dumps')
            if not os.path.isdir(_dd):
                os.makedirs(_dd)
            _ts = time.strftime('%m%d_%H%M%S')
            dump_path = os.path.join(_dd, '%s.pcm' % _ts)
            with open(os.path.join(_dd, '%s.txt' % _ts), 'w') as _tf:
                _tf.write(text)
            _all = sorted(x for x in os.listdir(_dd) if x.endswith('.pcm'))
            for _old in _all[:-300]:
                for _ext in ('.pcm', '.txt'):
                    try:
                        os.unlink(os.path.join(_dd, _old[:-4] + _ext))
                    except Exception:
                        pass
        except Exception:
            dump_path = None

        def _send(f32):
            nonlocal first, total
            if f32 is None or len(f32) == 0: return
            if first is None:
                first = time.time() - t0
                print('[tts] first audio %.2fs' % first, flush=True)
            data = (np.clip(f32, -1, 1) * 32767).astype('<i2').tobytes()
            total += len(f32)
            if dump_path:
                try:
                    with open(dump_path, 'ab') as _df:
                        _df.write(data)
                except Exception:
                    pass
            self.wfile.write(('%X\r\n' % len(data)).encode())
            self.wfile.write(data)
            self.wfile.write(b'\r\n')
            self.wfile.flush()

        def _apply_speed(x):
            """整段重采样。关键：必须整段做，逐块插值会让块边界相位跳变 -> 爆音。
            没 scipy 就用线性插值；降采样前先做个简单低通，压一压混叠。"""
            if abs(speed - 1.0) < 0.02 or len(x) < 64:
                return x
            src = x.astype(np.float32)
            if speed < 1.0:          # 变慢 = 插值放大，不需要抗混叠
                n = max(1, int(len(src) / speed))
                idx = np.linspace(0, len(src) - 1, n)
                return np.interp(idx, np.arange(len(src)), src).astype(np.float32)
            # 变快 = 降采样，先滑窗平均当低通，减少混叠
            k = max(1, int(round(speed)))
            if k > 1 and len(src) > k * 2:
                ker = np.ones(k, dtype=np.float32) / k
                src = np.convolve(src, ker, mode='same').astype(np.float32)
            n = max(1, int(len(src) / speed))
            idx = np.linspace(0, len(src) - 1, n)
            return np.interp(idx, np.arange(len(src)), src).astype(np.float32)

        def emit(f32):
            if f32 is None or len(f32) == 0:
                return
            seg_buf.append(np.asarray(f32, dtype=np.float32))

        def flush_seg():
            if not seg_buf:
                return None
            x = np.concatenate(seg_buf) if len(seg_buf) > 1 else seg_buf[0]
            seg_buf.clear()
            # 语速已经在码表中处理（tts_mobile.stretch_codes，保持音高），
            # 这里绝不能再对波形重采样 —— 那会连带改变音高。
            _send(_fade(x))
            return x

        aux_n = float(aux)
        seg_count = 0
        gapmul = float(gap)          # 气口倍率；别被下面段尾静音的局部变量覆盖掉
        SYNTH_LOCK.acquire()
        _pt_hold = [None]
        try:
            segs = plan_segments(text)
            if not segs:
                segs = [text]
            print('[tts] segs=%d %s' % (len(segs), [len(s) for s in segs]), flush=True)
            # ── 流水线并行 ──
            # 原来的顺序是：算完 seg0 的 s1s2，再解码 seg0，再算 seg1……
            # 现在让"算下一段"和"解码当前段"同时进行：
            #   生产者线程：s1s2 → 队列
            #   主线程：从队列取 → dec → 发送（保持顺序）
            import queue as _queue
            import threading as _threading
            _pipe = _queue.Queue(maxsize=2)
            _perr = []

            def _producer():
                try:
                    for _seg in segs:
                        _t = time.time()
                        _z, _c, _n = ENG.synth(_seg, ref_id=ref_id, speed=speed, a=aux_n)
                        _pipe.put((_seg, _z, _c, time.time() - _t))
                except Exception as _e:
                    _perr.append(_e)
                finally:
                    try:
                        _pipe.put(None)
                    except Exception:
                        pass

            _pt = _threading.Thread(target=_producer, daemon=True)
            _pt_hold[0] = _pt
            _pt.start()

            i = -1
            while True:
                _item = _pipe.get()
                if _item is None:
                    # 生产者已经退出。有异常就抛出来 ——
                    # 原来这里先 break 再查 _perr，永远不会执行到，
                    # 客户端会收到 HTTP 200 + 0 字节音频，日志只有一行 done。
                    if _perr:
                        raise _perr[0]
                    break
                i += 1
                seg, z, codes, tsyn = _item
                _td = time.time()
                with DEC_LOCK:
                    for chunk in dec_stream(z):
                        emit(chunk)
                _tdec = time.time() - _td
                _sent = flush_seg()
                if _sent is not None:
                    _z, _h = audio_metrics(_sent)
                    _cv = envelope_cv(_sent)
                    _wcv = min_window_cv(_sent)
                    if _cv < 0.28:
                        print('[tts] !! 平坦段(疑似嗡鸣) cv=%.2f zcr=%.2f hf6k=%.2f dur=%.2fs codes=%d len=%d txt=%s'
                              % (_cv, _z, _h, len(_sent) / float(SR), len(codes), len(seg), seg[:16]), flush=True)
                    else:
                        print('[tts] seg%d cv=%.2f wcv=%.2f zcr=%.2f hf6k=%.2f dur=%.2fs codes=%d'
                              % (i, _cv, _wcv, _z, _h, len(_sent) / float(SR), len(codes)), flush=True)
                    if _wcv < 0.15 and len(_sent) > SR:
                        print('[tts] !! 局部平坦(疑似嗡鸣) wcv=%.2f cv=%.2f dur=%.2fs codes=%d txt=%s'
                              % (_wcv, _cv, len(_sent) / float(SR), len(codes), seg[:16]), flush=True)
                    if _h > 0.45 or _z > 0.45:
                        print('[tts] !! 疑似噪声段 zcr=%.2f hf6k=%.2f dur=%.2fs codes=%d s=%.2f a=%.0f ref=%s'
                              % (_z, _h, len(_sent) / float(SR), len(codes), speed, aux_n, ref_id), flush=True)
                # 段间静音：段尾只补一次（静音不参与变速）。
                # 这里原来还 emit 了一份进 seg_buf —— 它会在下一次 flush 时挂到
                # 下一段的开头，既让静音翻倍，又让下一段的淡入全作用在静音上，
                # 真正的语音起点变成硬切。
                if i < len(segs) - 1:
                    gs = max(0.0, float(trailing_gap(seg)) * gapmul)
                    if gs > 0.01:
                        _send(np.zeros(int(SR * gs), dtype=np.float32))
                seg_count += 1
                print('[tts] seg%d len=%d codes=%d s1s2=%.2fs dec=%.2fs' % (i, len(seg), len(codes), tsyn, _tdec), flush=True)
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()
            print('[tts] done segs=%d audio=%.2fs front=%.2fs total=%.2fs' % (
                seg_count, total/SR, first or -1, time.time()-t0), flush=True)
        except Exception as e:
            print('[tts] ERR', type(e).__name__, e, flush=True)
            try: self.wfile.write(b'0\r\n\r\n'); self.wfile.flush()
            except Exception: pass
        finally:
            # 先等生产者退出再放锁：否则它可能还在 ENG.synth 里跑，
            # 下一个请求会和它并发进同一个引擎（内存翻倍），
            # 而它随后的 put 会永久阻塞在 maxsize=2 的队列上。
            _t = _pt_hold[0]
            if _t is not None and _t.is_alive():
                try:
                    _t.join(timeout=10.0)
                except Exception:
                    pass
            SYNTH_LOCK.release()

def keep_warm():
    """闲置太久就把模型页重新焐热。

    手机内存紧张时，这块内存会被 zram 换出去，下次开口就要先解压，
    听感就是「放一会儿之后她说话巨慢」。这里在完全没人用的时候
    每 4 分钟跑一次极短的合成，把页留住、CPU 也不会掉到最低频。
    """
    import urllib.request as _u
    while True:
        time.sleep(60)
        if time.time() - LAST_REQ[0] < 240:
            continue
        try:
            body = json.dumps({"text": "嗯", "ref": ENG.default_ref, "a": 0, "s": 1.0, "g": 1.0}).encode()
            req = _u.Request("http://127.0.0.1:8080/tts", data=body,
                             headers={"Content-Type": "application/json"})
            _u.urlopen(req, timeout=120).read()
            print("[app] keep-warm done", flush=True)
        except Exception as e:
            print("[app] keep-warm failed: %s" % e, flush=True)


def main():
    global ENG
    print('[app] loading engine...', flush=True)
    # top_k 原来是 8：长句子里模型分布比较平，只留 8 个候选会让它反复挑同几个，
    # 生成出来的 code 段几乎不变，解码就是那截嗡嗡。官方 GPT-SoVITS 用的是 15。
    _ref = os.environ.get('KATTS_REF_NPZ') or os.path.join(_HERE, 'refs', 'default.npz')
    ENG = tts_mobile.MobileTTS(models_dir=os.path.join(_HERE, 'models'),
                               ref_npz=_ref if os.path.exists(_ref) else None,
                               top_k=16)
    print('[app] engine ready', flush=True)
    start_dec()
    print('[app] warming up...', flush=True)
    eng_synth = ENG.synth("你好。")
    print('[app] warm ok, z=%s' % (eng_synth[0].shape,), flush=True)
    import threading
    threading.Thread(target=keep_warm, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', 8080), H)
    print('[app] listening http://127.0.0.1:8080', flush=True)
    srv.serve_forever()

if __name__ == '__main__':
    main()