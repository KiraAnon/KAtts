// dec_pipeline.cpp —— 在单个进程内串联 dec 的 11 个分段（CPU+APU 混合）
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <memory>

using namespace MNN;

static double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static bool readBin(const char* p, std::vector<float>& v) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end); std::streamoff n = f.tellg(); f.seekg(0);
    v.resize((size_t)n / sizeof(float));
    f.read((char*)v.data(), n);
    return true;
}

struct SegDef {
    const char* file;       // 模型文件名
    const char* inTensor;   // 唯一真正需要的输入张量名（nullptr 表示用 z/ge）
    const char* outTensor;
    bool apu;
};

static const SegDef SEGS[] = {
    {"s0_cpu.mnn",  nullptr,                              "/dec/ups.0/ConvTranspose_output_0", false},
    {"s1_apu.mnn",  "/dec/ups.0/ConvTranspose_output_0",  "/dec/Add_2_output_0",               true},
    {"s2_cpu.mnn",  "/dec/Add_2_output_0",                "/dec/ups.1/ConvTranspose_output_0", false},
    {"s3_apu.mnn",  "/dec/ups.1/ConvTranspose_output_0",  "/dec/Add_4_output_0",               true},
    {"s4_cpu.mnn",  "/dec/Add_4_output_0",                "/dec/ups.2/ConvTranspose_output_0", false},
    {"s5_apu.mnn",  "/dec/ups.2/ConvTranspose_output_0",  "/dec/Add_6_output_0",               true},
    {"s6_cpu.mnn",  "/dec/Add_6_output_0",                "/dec/ups.3/ConvTranspose_output_0", false},
    {"s7_apu.mnn",  "/dec/ups.3/ConvTranspose_output_0",  "/dec/Add_8_output_0",               true},
    {"s8_cpu.mnn",  "/dec/Add_8_output_0",                "/dec/ups.4/ConvTranspose_output_0", false},
    {"s9_apu.mnn",  "/dec/ups.4/ConvTranspose_output_0",  "/dec/Add_10_output_0",              true},
    {"s10_cpu.mnn", "/dec/Add_10_output_0",               "audio",                             false},
};
static const int NSEG = sizeof(SEGS) / sizeof(SEGS[0]);

struct Seg {
    std::shared_ptr<Interpreter> net;
    Session* sess = nullptr;
};

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("usage: dec_pipeline <modelDir> <z.bin> <ge.bin> [out.bin] [T=64]\n");
        return 1;
    }
    std::string dir = argv[1];
    if (!dir.empty() && dir.back() != '/') dir += "/";
    const char* outPath = argc > 4 ? argv[4] : nullptr;
    int T = argc > 5 ? atoi(argv[5]) : 64;

    std::vector<float> z, ge;
    if (!readBin(argv[2], z) || !readBin(argv[3], ge)) { printf("ERR read input\n"); return 1; }
    printf("[in] z=%zu floats, ge=%zu floats, T=%d\n", z.size(), ge.size(), T);

    ScheduleConfig cfgCPU; cfgCPU.type = MNN_FORWARD_CPU; cfgCPU.numThread = 4;
    BackendConfig bcA; bcA.precision = BackendConfig::Precision_Low; bcA.power = BackendConfig::Power_High;
    ScheduleConfig cfgAPU; cfgAPU.type = MNN_FORWARD_NN; cfgAPU.numThread = 4; cfgAPU.backendConfig = &bcA;

    std::vector<Seg> segs(NSEG);
    std::vector<double> tCreate(NSEG, 0), tRun(NSEG, 0);
    double t0 = nowMs();
    for (int i = 0; i < NSEG; ++i) {
        std::string path = dir + SEGS[i].file;
        segs[i].net.reset(Interpreter::createFromFile(path.c_str()));
        if (!segs[i].net) { printf("ERR load %s\n", path.c_str()); return 1; }
        double s = nowMs();
        segs[i].sess = segs[i].net->createSession(SEGS[i].apu ? cfgAPU : cfgCPU);
        if (!segs[i].sess) { printf("ERR createSession %s\n", SEGS[i].file); return 1; }
        segs[i].net->resizeSession(segs[i].sess);   // 固定 shape 也要 resize 一次，否则内存未按 shape 分配
        tCreate[i] = nowMs() - s;
    }
    printf("[init] 11 段加载+建会话 %.1f ms\n", nowMs() - t0);

    // 第一段：喂 z / ge
    auto runFirst = [&]() -> bool {
        auto& s = segs[0];
        Tensor* tz = s.net->getSessionInput(s.sess, "z");
        Tensor* tg = s.net->getSessionInput(s.sess, "ge");
        if (tz) {
            if (tz->elementSize() != (int)z.size()) { printf("ERR z size %d vs %zu\n", tz->elementSize(), z.size()); return false; }
            auto h = Tensor::create<float>(tz->shape(), nullptr, Tensor::CAFFE);
            memcpy(h->host<float>(), z.data(), z.size() * sizeof(float));
            tz->copyFromHostTensor(h); Tensor::destroy(h);
        }
        if (tg) {
            auto h = Tensor::create<float>(tg->shape(), nullptr, Tensor::CAFFE);
            memcpy(h->host<float>(), ge.data(), ge.size() * sizeof(float));
            tg->copyFromHostTensor(h); Tensor::destroy(h);
        }
        double st = nowMs();
        s.net->runSession(s.sess);
        tRun[0] = nowMs() - st;
        return true;
    };

    // 取某段输出到 host
    auto fetchOut = [&](int i, std::vector<float>& buf) -> bool {
        Tensor* o = segs[i].net->getSessionOutput(segs[i].sess, SEGS[i].outTensor);
        if (!o) { printf("ERR output %s of %s\n", SEGS[i].outTensor, SEGS[i].file); return false; }
        auto h = Tensor::create<float>(o->shape(), nullptr, Tensor::CAFFE);
        if (!o->copyToHostTensor(h)) { Tensor::destroy(h); return false; }
        size_t n = (size_t)h->elementSize();
        buf.assign(h->host<float>(), h->host<float>() + n);
        Tensor::destroy(h);
        return true;
    };

    // 把数据喂给第 i+1 段的指定输入
    auto feedNext = [&](int next, const std::vector<float>& buf) -> bool {
        Tensor* t = segs[next].net->getSessionInput(segs[next].sess, SEGS[next].inTensor);
        if (!t) { printf("ERR input %s of %s\n", SEGS[next].inTensor, SEGS[next].file); return false; }
        if (t->elementSize() != (int)buf.size()) {
            printf("ERR size mismatch %s: tensor %d vs data %zu\n", SEGS[next].inTensor, t->elementSize(), buf.size());
            return false;
        }
        auto h = Tensor::create<float>(t->shape(), nullptr, Tensor::CAFFE);
        memcpy(h->host<float>(), buf.data(), buf.size() * sizeof(float));
        bool ok = t->copyFromHostTensor(h);
        Tensor::destroy(h);
        return ok;
    };

    // ---- 流式模式：z 是长序列，内部按块处理 ----
    int chunk = 64, ctx = 0, drop = 0;
    for (int k = 0; k < argc; ++k) {
        if (!strcmp(argv[k], "--chunk") && k + 1 < argc) chunk = atoi(argv[k+1]);
        if (!strcmp(argv[k], "--ctx")   && k + 1 < argc) ctx   = atoi(argv[k+1]);
    }
    int Ttotal = (int)(z.size() / 192);
    if (Ttotal > chunk) {
        printf("[stream] z 共 %d 帧，按 %d 帧分块，左上下文 %d 帧\n", Ttotal, chunk, ctx);
        std::vector<float> all;
        const int step = std::max(1, chunk - ctx);   // 带上下文时每块只能推进 step 帧
        for (int start = 0; start < Ttotal; start += step) {
            int chStart = std::max(0, start - ctx);              // 含上下文的起点
            int skipFrames = start - chStart;                    // 左上下文帧数
            int core = std::min(step, Ttotal - start);           // 每块固定推进 step 帧，避免重叠
            if (core <= 0) break;
            int chLen = skipFrames + core;
            std::vector<float> zc((size_t)192 * chunk, 0.0f);
            for (int c = 0; c < 192; ++c)
                for (int t = 0; t < chLen && t < chunk; ++t)
                    zc[(size_t)c * chunk + t] = z[(size_t)c * Ttotal + chStart + t];

            // 跑一遍
            Tensor* tz = segs[0].net->getSessionInput(segs[0].sess, "z");
            auto hz = Tensor::create<float>(tz->shape(), nullptr, Tensor::CAFFE);
            memcpy(hz->host<float>(), zc.data(), zc.size() * sizeof(float));
            tz->copyFromHostTensor(hz); Tensor::destroy(hz);
            Tensor* tg = segs[0].net->getSessionInput(segs[0].sess, "ge");
            if (tg) { auto hg = Tensor::create<float>(tg->shape(), nullptr, Tensor::CAFFE);
                      memcpy(hg->host<float>(), ge.data(), ge.size()*sizeof(float));
                      tg->copyFromHostTensor(hg); Tensor::destroy(hg); }
            double st0 = nowMs();
            segs[0].net->runSession(segs[0].sess);
            std::vector<float> b;
            if (!fetchOut(0, b)) return 1;
            for (int i = 1; i < NSEG; ++i) {
                if (!feedNext(i, b)) return 1;
                segs[i].net->runSession(segs[i].sess);
                if (!fetchOut(i, b)) return 1;
            }
            double el = nowMs() - st0;
            int skip = skipFrames * 640;                         // 丢弃左上下文对应的音频
            int keep = core * 640;
            if (skip + keep > (int)b.size()) keep = (int)b.size() - skip;
            all.insert(all.end(), b.begin() + skip, b.begin() + skip + keep);
            printf("  chunk start=%3d core=%2d -> %.2fs  %.1f ms\n", start, core, keep/32000.0, el);
        }
        printf("[stream] 合计 %.2f s 音频\n", all.size()/32000.0);
        if (outPath) { std::ofstream f(outPath, std::ios::binary);
                       f.write((const char*)all.data(), all.size()*sizeof(float));
                       printf("[dump] %s\n", outPath); }
        return 0;
    }

    int repeat = 1;
    for (int k = 0; k < argc; ++k) if (!strcmp(argv[k], "--repeat") && k + 1 < argc) repeat = atoi(argv[k+1]);

    std::vector<float> buf, first;
    for (int rep = 0; rep < repeat; ++rep) {
        if (!runFirst()) return 1;
        for (int i = 0; i < NSEG; ++i) {
            if (i > 0) {
                double st = nowMs();
                segs[i].net->runSession(segs[i].sess);
                tRun[i] = nowMs() - st;
            }
            if (!fetchOut(i, buf)) return 1;
            if (i + 1 < NSEG && !feedNext(i + 1, buf)) return 1;
        }
        if (rep == 0) first = buf;
        else {
            double md = 0; size_t n = std::min(first.size(), buf.size());
            for (size_t j = 0; j < n; ++j) md = std::max(md, (double)std::fabs(first[j] - buf[j]));
            printf("[repeat %d] 与首轮最大差异 = %.3e\n", rep, md);
        }
    }

    printf("[seg] ");
    double sum = 0;
    for (int i = 0; i < NSEG; ++i) {
        printf("%.1f ", tRun[i]);
        sum += tRun[i];
    }
    printf("\n[perf] 总耗时 %.2f ms（含首段）\n", sum);

    printf("[out] audio %zu floats -> %.2f s\n", buf.size(), buf.size() / 32000.0);
    if (outPath) {
        std::ofstream f(outPath, std::ios::binary);
        f.write((const char*)buf.data(), buf.size() * sizeof(float));
        printf("[dump] %s\n", outPath);
    }
    return 0;
}