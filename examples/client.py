"""最小调用示例：合成一句话并保存为 wav。"""
import json
import urllib.request
import wave

SERVER = "http://127.0.0.1:8080"


def synth(text, ref=None, speed=1.0, aux=0.0, gap=1.0):
    body = {"text": text, "s": speed, "a": aux, "g": gap}
    if ref:
        body["ref"] = ref
    req = urllib.request.Request(
        SERVER,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    pcm = bytearray()
    with urllib.request.urlopen(req, timeout=300) as r:
        while True:
            chunk = r.read(8192)
            if not chunk:
                break
            pcm += chunk
    return bytes(pcm)


def save_wav(path, pcm, sr=32000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


if __name__ == "__main__":
    data = synth("你好，我是本地合成的语音。")
    save_wav("out.wav", data)
    print("saved out.wav  %.2f s" % (len(data) / 2 / 32000))
