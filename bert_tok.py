import json, unicodedata

import os
_V = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert_vocab.json'), encoding='utf-8'))
CLS, SEP, UNK = 101, 102, 100

def _is_punct(ch):
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith('P')

def _is_cjk(ch):
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0xF900 <= cp <= 0xFAFF)

def _clean(text):
    out = []
    for ch in text:
        cp = ord(ch)
        if cp == 0 or cp == 0xFFFD: continue
        if cp < 32 or cp == 127: continue
        if (cp < 128) and chr(cp).isspace():
            out.append(' ')
        elif unicodedata.category(ch) == 'Zs':
            out.append(' ')
        else:
            out.append(ch)
    return ''.join(out)

def _pre_tokenize(text):
    words, cur = [], ''
    for ch in text:
        if ch.isspace():
            if cur: words.append(cur); cur = ''
        elif _is_cjk(ch):
            if cur: words.append(cur); cur = ''
            words.append(ch)
        elif _is_punct(ch):
            if cur: words.append(cur); cur = ''
            words.append(ch)
        else:
            cur += ch
    if cur: words.append(cur)
    return words

def _wordpiece(word):
    if not word: return []
    if word in _V: return [_V[word]]
    ids, start = [], 0
    while start < len(word):
        end = len(word); cur = None
        while start < end:
            sub = word[start:end]
            cand = sub if start == 0 else '##' + sub
            if cand in _V:
                cur = _V[cand]; break
            end -= 1
        if cur is None:
            return [UNK]
        ids.append(cur); start = end
    return ids

def tokenize(text):
    text = _clean(text).lower()
    ids = [CLS]
    for w in _pre_tokenize(text):
        ids.extend(_wordpiece(w))
    ids.append(SEP)
    return ids, [1]*len(ids)

if __name__ == '__main__':
    for t in ['今天天气很好，我终于把这个东西跑通了。','他说 OK 就行了，明天见。','GPT-SoVITS 是个好项目 2026 年。','Hello world, 这是测试。']:
        ids,_ = tokenize(t)
        print('%-28s -> %d  %s' % (t[:26], len(ids), ids[:12]), flush=True)