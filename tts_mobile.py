"""手机端 TTS 推理引擎（纯 numpy + onnxruntime，无 torch）

链路：文本 -> phones -> bert -> s1 -> codes -> s2_pre -> z -> dec_pipeline -> 音频
"""
import os, sys, math, json, time
import numpy as np
import onnxruntime as ort

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault('is_g2pw', '0')

from text.cleaner import clean_text
from text import cleaned_text_to_sequence
import bert_tok

EOS = 1024
N_LAYER = 24


def make_sess(path, threads=2, arena=True):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    # 每句只跑一次的模型（BERT）关掉内存池：它一上来就预留一大块，
    # 而这台机器内存本来就紧。热路径（s1_step）仍然开池，保证速度。
    so.enable_cpu_mem_arena = arena
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])


class MobileTTS:
    def __init__(self, models_dir=None, ref_npz=None, top_k=8):
        self.models = models_dir or os.path.join(_HERE, 'models')
        # 逐文件自检：缺哪个直接说哪个，别让它抛一堆 FileNotFoundError 堆栈
        _need = ['s1_params.npz', 's1_prefill_mask18.onnx', 's1_step_int8.onnx',
                 's2_pre.onnx', 'bert_int8.onnx']
        _miss = [f for f in _need if not os.path.exists(os.path.join(self.models, f))]
        if _miss:
            raise RuntimeError(
                '缺少模型文件：%s' % ', '.join(_miss)
                + chr(10) + '目录：' + self.models
                + chr(10) + '下载方法见 models/README.md')
        self.top_k = top_k
        p = np.load(os.path.join(self.models, 's1_params.npz'))
        self.TE = p['ar_text_embedding']
        self.AE = p['ar_audio_embedding']
        self.BPW, self.BPB = p['bert_proj_w'], p['bert_proj_b']
        self.PRW, self.PRB = p['predict_w'], p['predict_b']
        self.AT = float(p['ar_text_position_alpha'][0])
        self.AA = float(p['ar_audio_position_alpha'][0])
        self._div = np.exp(np.arange(0, 512, 2, dtype=np.float64) * -(math.log(10000.0) / 512))
        self._pe = {}
        self.prefill = make_sess(os.path.join(self.models, 's1_prefill_mask18.onnx'))
        self.step = make_sess(os.path.join(self.models, 's1_step_int8.onnx'))
        self.s2 = make_sess(os.path.join(self.models, 's2_pre.onnx'), arena=False)
        self.bert = make_sess(os.path.join(self.models, 'bert_int8.onnx'), arena=False)
        self.refs = {}
        # 音色目录：就在本文件同级的 refs/ 下。
        # （原来按 ref_npz 的父目录再拼 'refs'，ref_npz 一旦放进 refs/ 就会变成 refs/refs）
        refs_dir = os.path.join(_HERE, 'refs')
        if os.path.isdir(refs_dir):
            for fn in sorted(os.listdir(refs_dir)):
                if not fn.endswith('.npz'):
                    continue
                r = np.load(os.path.join(refs_dir, fn))
                self.refs[fn[:-4]] = {
                    'ph': r['p_phones'].reshape(-1).astype(np.int64),
                    'bf': r['p_bert_features'].astype(np.float32),
                    'sem': r['p_prompt_semantic'].reshape(1, -1).astype(np.int64),
                    'ge': r['ge'].astype(np.float32),
                }
        # 兜底：单个参考音色文件（可选，优先级低于 refs/ 目录）
        if ref_npz and os.path.exists(ref_npz):
            _base = np.load(ref_npz)
            self.refs.setdefault('default', {
                'ph': _base['p_phones'].reshape(-1).astype(np.int64),
                'bf': _base['p_bert_features'].astype(np.float32),
                'sem': _base['p_prompt_semantic'].reshape(1, -1).astype(np.int64),
                'ge': _base['ge'].astype(np.float32),
            })
        if not self.refs:
            raise RuntimeError(
                '找不到任何参考音色。请把 GPT-SoVITS 提取的音色特征（*.npz）'
                '放到 %s/ 下，生成方法见 refs/README.md' % refs_dir)
        _want = os.environ.get('KATTS_DEFAULT_REF', '').strip()
        if _want in self.refs:
            self.default_ref = _want
        else:
            # 显式优先中文音色。原来取 sorted()[0]，会落到 ja_concern（日语音色 + 关切情绪），
            # 中文用户一开口就是日语味。
            for _pref in ('zh_normal', 'zh_warm', 'default', 'ja_warm'):
                if _pref in self.refs:
                    self.default_ref = _pref
                    break
            else:
                self.default_ref = sorted(self.refs)[0]
        self.ge_ = self.refs[self.default_ref]['ge']

    def _pos(self, idx):
        # 一次建好足够大的表，之后只做切片（原来每步都重建，白白烧掉大量时间）
        CAP = 4096
        if 'tbl' not in self._pe:
            p = np.arange(0, CAP, dtype=np.float64)[:, None]
            e = np.zeros((CAP, 512))
            e[:, 0::2] = np.sin(p * self._div)
            e[:, 1::2] = np.cos(p * self._div)
            self._pe['tbl'] = e
        return self._pe['tbl'][idx]

    def text_to_phones(self, text):
        if text and text[-1] not in '。！？，.!,?':
            text = text + '。'
        # 第三个参数是 version（'v1'/'v2'），不是语言。
        # 原来写的是 'zh' —— 只因为不等于 'v1' 才碰巧走了 v2 分支，属于运气。
        # 参考音频那边是 prompt_lang='ja'（见 refs/build_refs.py），
        # 所以这里是标准的跨语言合成：中文文本 + 日语音色。
        ph, w2p, norm = clean_text(text, 'zh', 'v2')
        return cleaned_text_to_sequence(ph), w2p, norm

    def bert_feature(self, text, word2ph):
        ids, am = bert_tok.tokenize(text)
        ids = np.array([ids], dtype=np.int64)
        feed = {'input_ids': ids,
                'attention_mask': np.ones_like(ids, dtype=np.int64),
                'token_type_ids': np.zeros_like(ids, dtype=np.int64)}
        out = self.bert.run(None, feed)[0][0]
        res = out[1:-1]
        if len(word2ph) != res.shape[0]:
            raise ValueError('bert 长度不匹配: word2ph=%d tokens=%d' % (len(word2ph), res.shape[0]))
        feats = [res[i][None, :].repeat(int(word2ph[i]), 0) for i in range(len(word2ph))]
        return np.concatenate(feats, axis=0).T.astype(np.float32)

    def _mask(self, Lx, Ly, H=16):
        xa = np.zeros((Lx, Lx), dtype=bool)
        xa_pad = np.pad(xa, ((0, 0), (0, Ly)), constant_values=True)
        ya = np.triu(np.ones((Ly, Ly), dtype=bool), k=1)
        ya_pad = np.pad(ya, ((0, 0), (Lx, 0)), constant_values=False)
        m2 = np.concatenate([xa_pad, ya_pad], axis=0)
        return np.broadcast_to(m2, (1, H, m2.shape[0], m2.shape[1])).copy()

    @staticmethod
    def _softmax(x):
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    # 重复惩罚只看最近这么多个 token
    REP_WINDOW = 20

    def _pick(self, lg, prev, rng):
        x = lg[0].astype(np.float64)
        # 重复惩罚只应该看"刚才说过什么"。
        # 原来对全部已生成的 token 惩罚：句子越长，被压掉 logit 的 token 越多，
        # 到后半句可选空间被挤得很窄，模型只能在那几个里反复挑 ——
        # 结果就是长句中间某一段变成又平又直的嗡嗡（实测 cv 从 0.9 掉到 0.13）。
        recent = prev[-self.REP_WINDOW:] if len(prev) > self.REP_WINDOW else prev
        if len(recent):
            V = x.shape[-1]
            seen = np.zeros(V, dtype=bool)
            arr = np.asarray(recent, dtype=np.int64)
            arr = arr[(arr >= 0) & (arr < V)]
            seen[arr] = True
            idx = np.flatnonzero(seen)
            if len(idx):
                sel = x[idx]
                x[idx] = np.where(sel < 0, sel * 1.35, sel / 1.35)
        k = self.top_k if self.top_k else 64
        k = min(k, x.shape[-1])
        idx = np.argpartition(-x, k - 1)[:k]
        vals = x[idx]
        vals = vals - vals.max()
        pr = np.exp(vals)
        pr = pr / pr.sum()
        q = rng.exponential(1.0, size=k)
        return int(idx[int(np.argmax(pr / q))])

    def synth_codes(self, phones, bf, seed=1234, max_steps=600, ref=None, limit=None):
        ref = ref or self.refs[self.default_ref]
        ph = np.concatenate([ref['ph'], np.asarray(phones, dtype=np.int64)])[None]
        b = np.concatenate([ref['bf'], np.asarray(bf, dtype=np.float32)], axis=1)[None]
        prompt = ref['sem']
        Lx, Ly = ph.shape[1], prompt.shape[1]
        x = self.TE[ph] + (b.transpose(0, 2, 1) @ self.BPW.T + self.BPB) + self.AT * self._pos(np.arange(Lx))[None]
        y0 = self.AE[prompt] + self.AA * self._pos(np.arange(Ly))[None]
        xy = np.concatenate([x, y0], axis=1).astype(np.float32)
        out, k, v = self.prefill.run(None, {'x': xy, 'attn_mask': self._mask(Lx, Ly)})
        rng = np.random.default_rng(seed)
        y = list(prompt[0])
        # 长度上限：正常大概 2~2.5 个 codes/音素。超过 4 倍还不停，
        # 就是模型开始复读了 —— 再往下合成的就是那串电子震动音。
        if limit is None:
            limit = max(24, int(len(phones) * 4))
        for i in range(max_steps):
            lg = out[:, -1] @ self.PRW.T + self.PRB
            if i < 11:
                lg = lg[:, :-1]
            tok = self._pick(lg, y, rng)
            if i >= 11 and tok == EOS:
                # 正常一段语音大约 2~2.5 个 semantic code / 音素。
                # EOS 来的时候如果只生成了很少的 code，说明它"提前收尾"了：
                # 这些 code 交给 s2 会把它们强行拉长铺满整句，
                # 出来就是那股又平又长、没有快慢的电子嗡鸣。
                # 所以：太早的 EOS 不放行，把最后一维（EOS）切掉重新抽一个。
                if (len(y) - Ly) >= max(12, int(len(phones) * 1.8)):
                    self.last_exit = 'eos'
                    break
                self.last_exit = 'eos_early'
                print('[tts] EOS 来得太早 codes=%d phones=%d，屏蔽后继续'
                      % (len(y) - Ly, len(phones)), flush=True)
                tok = self._pick(lg[:, :-1], y, rng)
            if i >= 11 and (len(y) - Ly) >= limit:
                print('[tts] codes 顶到上限 %d (phones=%d)，停止生成' % (limit, len(phones)), flush=True)
                self.last_exit = 'limit'
                break
            y.append(tok)
            ye = (self.AE[[[tok]]] + self.AA * self._pos(np.array([Ly + i]))[None]).astype(np.float32)
            out, k, v = self.step.run(None, {'x': ye, 'k_cache': k, 'v_cache': v})
        else:
            self.last_exit = 'steps'
        return np.array(y[Ly:], dtype=np.int64)

    def codes_to_z(self, codes, text_ids, ge):
        c = codes.reshape(1, 1, -1).astype(np.int64)
        return self.s2.run(None, {'codes': c, 'text': text_ids, 'ge': ge})[0].astype(np.float32)
    @property
    def ge(self):
        return self.ge_

    @staticmethod
    def stretch_codes(codes, speed):
        """按 speed 缩放 semantic codes 的时间轴（保持音高）。

        GPT-SoVITS 的 speed_factor 作用在 decode_encp 的 y_lengths 上
        （sizee = codes_len * 4 / speed），也就是把语义序列按 1/speed
        拉长或压缩之后再解码 —— 变的只有每个音节的持续时长，音高不动。
        之前手机端是对合成好的波形做重采样，那是变速变调：
        s>1 整段升调（听起来像小孩），s<1 整段降调。"""
        if abs(speed - 1.0) < 0.01 or len(codes) < 2:
            return codes
        n = max(1, int(round(len(codes) / speed)))
        idx = np.linspace(0, len(codes) - 1, n)
        return codes[np.round(idx).astype(np.int64)]

    def mix_ge(self, name, a, anchor=None):
        """a 轴：把锚点参考和目标参考的 ge 按票数平均。

        多参考就是 torch.stack(ges).mean(0)：
        锚点固定 1 票，目标参考重复 a 次（a 票）。
        a=0 -> 纯锚点音色（原来是写死的 ja_warm，现在跟默认音色走）；
        a 越大越偏向目标参考的音色。
        """
        if anchor is None:
            anchor = os.environ.get('KATTS_ANCHOR_REF', '').strip() or self.default_ref
        if anchor not in self.refs or name == anchor:
            return self.refs[name]['ge']
        n = int(round(float(a or 0)))
        if n <= 0:
            return self.refs[anchor]['ge']
        stack = [self.refs[anchor]['ge'].astype(np.float32)]
        stack += [self.refs[name]['ge'].astype(np.float32)] * n
        return np.stack(stack, 0).mean(0)

    @staticmethod
    def _degenerate(c):
        """判断这一轮 codes 是不是"坍缩"了。

        整句变成平坦嗡嗡声，就是 s1 吐出的 codes 几乎没有变化：
        不同值太少，或者同一段连续重复太长。正常语音 codes 变化很密，
        不同值通常占 40% 以上。
        """
        n = len(c)
        if n < 10:
            return True
        try:
            uniq = len(set(int(x) for x in c))
        except Exception:
            return False
        if uniq < max(5, int(n * 0.30)):
            return True
        maxrun, cur = 1, 1
        for i in range(1, n):
            if c[i] == c[i - 1]:
                cur += 1
                if cur > maxrun:
                    maxrun = cur
            else:
                cur = 1
        if maxrun > n * 0.5:
            return True
        # 局部塌缩：中间某一小段几乎不变（整段统计看不出来的那种）
        if MobileTTS._local_collapse(c) < 0.4:
            return True
        # 尾部塌缩：句尾那个字变成嗡嗡，就是最后一段 codes 几乎不变
        return MobileTTS._tail_bad(c)

    @staticmethod
    def _local_collapse(c, win=12):
        """滑动窗口找最"死"的一小段。

        整句 uniq 比例正常、也没有超长重复，但中间某一小段几乎不变 ——
        解码出来就是那一截嗡鸣。整段统计看不出来，必须按窗口看。
        返回最差窗口的 uniq 比例（正常 0.9+，坍缩会掉到 0.4 以下）。
        """
        n = len(c)
        if n < win * 2:
            return 1.0
        worst = 1.0
        step = max(1, win // 3)
        for i in range(0, n - win + 1, step):
            w = c[i:i + win]
            try:
                r = len(set(int(x) for x in w)) / float(win)
            except Exception:
                continue
            if r < worst:
                worst = r
        return worst

    @staticmethod
    def _tail_bad(c):
        n = len(c)
        if n < 12:
            return False
        k = max(6, n // 4)
        tail = c[-k:]
        try:
            uniq = len(set(int(x) for x in tail))
        except Exception:
            return False
        if uniq < max(3, int(k * 0.30)):
            return True
        run, cur = 1, 1
        for i in range(1, len(tail)):
            if tail[i] == tail[i - 1]:
                cur += 1
                if cur > run:
                    run = cur
            else:
                cur = 1
        return run > len(tail) * 0.5

    @staticmethod
    def _cut_bad_tail(c):
        """从尾部往回切，直到尾部不再塌缩（最多切掉一半）。"""
        n = len(c)
        for cut in range(0, n // 2, 2):
            cand = c[:n - cut] if cut else c
            if len(cand) >= 12 and not MobileTTS._tail_bad(cand):
                return cand
        return c

    def synth(self, text, seed=1234, ref_id=None, speed=1.0, a=0):
        """返回 (z, codes, norm_text)；z 是 [1,192,T]，交给 dec 出音频"""
        name = ref_id if (ref_id and ref_id in self.refs) else self.default_ref
        ref = self.refs[name]
        _t0 = time.time()
        phones, w2p, norm = self.text_to_phones(text)
        _t1 = time.time()
        bf = self.bert_feature(norm, w2p)
        _t2 = time.time()
        # 实测正常约 2.0~2.6 codes/音素。原来用 3 倍太紧：
        # 「好呀，我去找找看」16 个音素被卡到 48，三次重试全被截断，
        # 末尾直接没了 —— 听感就是被一刀切。
        limit = max(24, int(len(phones) * 8))
        codes = None
        bad_hits = 0
        for k, sd in enumerate((seed, seed + 977, seed + 1954)):
            c = self.synth_codes(phones, bf, seed=sd, ref=ref, limit=limit)
            try:
                uq = len(set(int(x) for x in c))
                try:
                    lc = self._local_collapse(c)
                except Exception:
                    lc = 1.0
                print('[dbg] try%d phones=%d codes=%d uniq=%d ratio=%.2f lw=%.2f exit=%s txt=%s'
                      % (k + 1, len(phones), len(c), uq, uq / max(1, len(c)), lc,
                         getattr(self, 'last_exit', '?'), text[:12]), flush=True)
            except Exception:
                pass
            if self._degenerate(c):
                bad_hits += 1
                print('[tts] codes 坍缩(第%d次) uniq=%d len=%d，换种子重试'
                      % (k + 1, len(set(int(x) for x in c)), len(c)), flush=True)
                continue                      # 这一轮不要
            if codes is None or len(c) < len(codes):
                codes = c
            if len(codes) < limit * 0.85:
                break
        if codes is None:
            # 三次都坍缩，退而求其次：用最后一次，并把尾部塌缩的那截切掉，
            # 否则句尾那个字会变成一声平直的嗡嗡。
            codes = self.synth_codes(phones, bf, seed=seed + 1954, ref=ref, limit=limit)
            codes = self._cut_bad_tail(codes)
            print('[tts] !! 三次全坍缩，已切掉尾部塌缩段 len=%d' % len(codes), flush=True)                       # 正常长度，不用再试
        if len(codes) >= limit:
            print('[tts] codes 三次都顶到上限(%d)，截断 text=%s'
                  % (limit, text[:24]), flush=True)
        _t3 = time.time()
        codes = self.stretch_codes(codes, speed)
        ge = self.mix_ge(name, a)
        z = self.codes_to_z(codes, np.asarray(phones, dtype=np.int64)[None], ge)
        _t4 = time.time()
        print('[prof] g2p=%.0fms bert=%.0fms s1=%.0fms s2=%.0fms phones=%d codes=%d'
              % ((_t1-_t0)*1000, (_t2-_t1)*1000, (_t3-_t2)*1000, (_t4-_t3)*1000,
                 len(phones), len(codes)), flush=True)
        # 诊断：z 是不是"后半段不动了"（s2 输出坍缩会让 dec 解出一段平音）
        try:
            T = z.shape[2]
            half = max(1, T // 2)
            dz = np.abs(np.diff(z[0], axis=1))
            d_all = float(dz.mean()) if dz.size else 0.0
            d_head = float(np.abs(np.diff(z[0][:, :half], axis=1)).mean()) if half > 1 else 0.0
            d_tail = float(np.abs(np.diff(z[0][:, half:], axis=1)).mean()) if T - half > 1 else 0.0
            print('[dbg] z T=%d diff=%.4f head=%.4f tail=%.4f'
                  % (T, d_all, d_head, d_tail), flush=True)
        except Exception as e:
            print('[dbg] z diag failed', e, flush=True)
        return z, codes, norm

    def write_wav(self, audio, path, sr=32000):
        import struct
        a = np.clip(audio, -1, 1)
        pcm = (a * 32767).astype(np.int16)
        with open(path, 'wb') as f:
            f.write(b'RIFF'); f.write(struct.pack('<I', 36 + pcm.nbytes))
            f.write(b'WAVEfmt '); f.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
            f.write(b'data'); f.write(struct.pack('<I', pcm.nbytes))
            f.write(pcm.tobytes())