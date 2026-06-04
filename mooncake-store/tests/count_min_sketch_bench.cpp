#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "count_min_sketch.h"
#include "atomic_count_min_sketch.h"

namespace mooncake {
namespace {

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

// A simple RAII timer returning elapsed milliseconds.
class Timer {
   public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    double elapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

   private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Generate |n| random string keys of the form "key_<uuid>".
std::vector<std::string> generateKeys(size_t n) {
    std::vector<std::string> keys;
    keys.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        keys.push_back("key_" + std::to_string(i));
    }
    return keys;
}

// ---------------------------------------------------------------------------
// Correctness: AtomicCountMinSketch behaves like CountMinSketch
// ---------------------------------------------------------------------------

TEST(CountMinSketchBench, AtomicCorrectnessBasic) {
    AtomicCountMinSketch sketch(64, 4);

    EXPECT_EQ(sketch.increment("key_a"), 1);
    EXPECT_EQ(sketch.increment("key_a"), 2);
    EXPECT_EQ(sketch.increment("key_a"), 3);
    EXPECT_EQ(sketch.increment("key_b"), 1);
    EXPECT_EQ(sketch.count("key_a"), 3);
    EXPECT_EQ(sketch.count("key_b"), 1);
    EXPECT_EQ(sketch.count("key_c"), 0);

    sketch.decay();
    EXPECT_EQ(sketch.count("key_a"), 1);
    EXPECT_EQ(sketch.count("key_b"), 0);
}

TEST(CountMinSketchBench, AtomicCorrectnessAutoDecay) {
    AtomicCountMinSketch sketch(8, 2);
    for (int i = 0; i < 15; ++i) {
        sketch.increment("hot_key");
    }
    EXPECT_EQ(sketch.count("hot_key"), 15);
    uint8_t ret = sketch.increment("hot_key");
    EXPECT_EQ(ret, 16);
    EXPECT_EQ(sketch.count("hot_key"), 8);
}

TEST(CountMinSketchBench, AtomicCorrectnessZeroDimensions) {
    AtomicCountMinSketch sketch(0, 0);
    EXPECT_EQ(sketch.count("zero_key"), 0);
    EXPECT_EQ(sketch.increment("zero_key"), 1);
    EXPECT_EQ(sketch.count("zero_key"), 1);
}

// ---------------------------------------------------------------------------
// Performance: single-threaded count() benchmark
// ---------------------------------------------------------------------------

TEST(CountMinSketchBench, SingleThreadCountPerformance) {
    constexpr size_t kNumKeys = 100000;
    constexpr size_t kRepeat = 100;
    auto keys = generateKeys(kNumKeys);

    // Pre-populate both sketches with the same data.
    CountMinSketch mutex_sketch(4096, 4);
    AtomicCountMinSketch atomic_sketch(4096, 4);
    for (const auto &k : keys) {
        mutex_sketch.increment(k);
        atomic_sketch.increment(k);
    }

    // ---- Benchmark mutex count() ----
    {
        Timer t;
        volatile uint8_t sink = 0;
        for (size_t r = 0; r < kRepeat; ++r) {
            for (const auto &k : keys) {
                sink = mutex_sketch.count(k);
            }
        }
        (void)sink;
        double ms = t.elapsedMs();
        std::cout << "[bench] Mutex   count() x " << kNumKeys * kRepeat
                  << " = " << ms << " ms  ("
                  << (kNumKeys * kRepeat / ms * 1000 / 1e6) << " Mops/s)"
                  << std::endl;
    }

    // ---- Benchmark atomic count() ----
    {
        Timer t;
        volatile uint8_t sink = 0;
        for (size_t r = 0; r < kRepeat; ++r) {
            for (const auto &k : keys) {
                sink = atomic_sketch.count(k);
            }
        }
        (void)sink;
        double ms = t.elapsedMs();
        std::cout << "[bench] Atomic  count() x " << kNumKeys * kRepeat
                  << " = " << ms << " ms  ("
                  << (kNumKeys * kRepeat / ms * 1000 / 1e6) << " Mops/s)"
                  << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Performance: single-threaded increment() benchmark
// ---------------------------------------------------------------------------

TEST(CountMinSketchBench, SingleThreadIncrementPerformance) {
    constexpr size_t kNumIncrements = 100000;
    auto keys = generateKeys(kNumIncrements);

    {
        CountMinSketch sketch(4096, 4);
        Timer t;
        for (const auto &k : keys) {
            sketch.increment(k);
        }
        double ms = t.elapsedMs();
        std::cout << "[bench] Mutex   increment() x " << kNumIncrements
                  << " = " << ms << " ms  ("
                  << (kNumIncrements / ms * 1000 / 1e6) << " Mops/s)"
                  << std::endl;
    }

    {
        AtomicCountMinSketch sketch(4096, 4);
        Timer t;
        for (const auto &k : keys) {
            sketch.increment(k);
        }
        double ms = t.elapsedMs();
        std::cout << "[bench] Atomic  increment() x " << kNumIncrements
                  << " = " << ms << " ms  ("
                  << (kNumIncrements / ms * 1000 / 1e6) << " Mops/s)"
                  << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Performance: multi-threaded count() benchmark (read-heavy)
// ---------------------------------------------------------------------------

template <typename Sketch>
void runMultiThreadedCount(Sketch &sketch, const std::vector<std::string> &keys,
                           size_t num_threads, size_t ops_per_thread,
                           const char *label) {
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;

    // Pre-warm keys into a per-thread random-access pattern
    std::vector<std::vector<std::string>> thread_keys(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        thread_keys[t].reserve(ops_per_thread);
        for (size_t i = 0; i < ops_per_thread; ++i) {
            thread_keys[t].push_back(keys[(t * ops_per_thread + i) % keys.size()]);
        }
    }

    Timer global_timer;
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            while (!start.load(std::memory_order_acquire)) {
            }
            volatile uint8_t sink = 0;
            for (const auto &k : thread_keys[t]) {
                sink = sketch.count(k);
            }
            (void)sink;
        });
    }

    start.store(true, std::memory_order_release);
    for (auto &th : threads) {
        th.join();
    }
    double ms = global_timer.elapsedMs();
    size_t total_ops = num_threads * ops_per_thread;
    std::cout << "[bench] " << label << " count() x " << total_ops
              << " (" << num_threads << " threads) = " << ms << " ms  ("
              << (total_ops / ms * 1000 / 1e6) << " Mops/s)" << std::endl;
}

TEST(CountMinSketchBench, MultiThreadCountPerformance) {
    constexpr size_t kNumKeys = 50000;
    constexpr size_t kNumThreads = 4;
    constexpr size_t kOpsPerThread = 250000;

    auto keys = generateKeys(kNumKeys);

    // Pre-populate both sketches identically.
    CountMinSketch mutex_sketch(4096, 4);
    AtomicCountMinSketch atomic_sketch(4096, 4);
    for (const auto &k : keys) {
        mutex_sketch.increment(k);
        atomic_sketch.increment(k);
    }

    runMultiThreadedCount(mutex_sketch, keys, kNumThreads, kOpsPerThread,
                          "Mutex  ");
    runMultiThreadedCount(atomic_sketch, keys, kNumThreads, kOpsPerThread,
                          "Atomic ");
}

// ---------------------------------------------------------------------------
// Performance: multi-threaded increment() benchmark (write-heavy)
// ---------------------------------------------------------------------------

template <typename Sketch>
void runMultiThreadedIncrement(Sketch &sketch,
                               const std::vector<std::string> &keys,
                               size_t num_threads, size_t ops_per_thread,
                               const char *label) {
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;

    std::vector<std::vector<std::string>> thread_keys(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        thread_keys[t].reserve(ops_per_thread);
        for (size_t i = 0; i < ops_per_thread; ++i) {
            thread_keys[t].push_back(keys[(t * ops_per_thread + i) % keys.size()]);
        }
    }

    Timer global_timer;
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            while (!start.load(std::memory_order_acquire)) {
            }
            for (const auto &k : thread_keys[t]) {
                sketch.increment(k);
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto &th : threads) {
        th.join();
    }
    double ms = global_timer.elapsedMs();
    size_t total_ops = num_threads * ops_per_thread;
    std::cout << "[bench] " << label << " increment() x " << total_ops
              << " (" << num_threads << " threads) = " << ms << " ms  ("
              << (total_ops / ms * 1000 / 1e6) << " Mops/s)" << std::endl;
}

TEST(CountMinSketchBench, MultiThreadIncrementPerformance) {
    constexpr size_t kNumKeys = 10000;
    constexpr size_t kNumThreads = 4;
    constexpr size_t kOpsPerThread = 50000;

    auto keys = generateKeys(kNumKeys);

    CountMinSketch mutex_sketch(4096, 4);
    AtomicCountMinSketch atomic_sketch(4096, 4);

    runMultiThreadedIncrement(mutex_sketch, keys, kNumThreads, kOpsPerThread,
                              "Mutex  ");
    runMultiThreadedIncrement(atomic_sketch, keys, kNumThreads, kOpsPerThread,
                              "Atomic ");
}

// ---------------------------------------------------------------------------
// Performance: mixed read/write workload (multi-threaded)
// ---------------------------------------------------------------------------

TEST(CountMinSketchBench, MultiThreadMixedPerformance) {
    constexpr size_t kNumKeys = 20000;
    constexpr size_t kNumThreads = 4;
    constexpr size_t kOpsPerThread = 100000;
    constexpr double kReadRatio = 0.9;  // 90% reads, 10% writes

    auto keys = generateKeys(kNumKeys);

    // Pre-populate
    CountMinSketch mutex_sketch(4096, 4);
    AtomicCountMinSketch atomic_sketch(4096, 4);
    for (const auto &k : keys) {
        mutex_sketch.increment(k);
        atomic_sketch.increment(k);
    }

    // Benchmark for mutex sketch
    {
        std::atomic<bool> start{false};
        std::vector<std::thread> threads;
        std::mt19937 rng(42);

        Timer global_timer;
        for (size_t t = 0; t < kNumThreads; ++t) {
            threads.emplace_back([&, t]() {
                std::mt19937 local_rng(42 + t);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                while (!start.load(std::memory_order_acquire)) {
                }
                volatile uint8_t sink = 0;
                for (size_t i = 0; i < kOpsPerThread; ++i) {
                    const auto &k = keys[(t * kOpsPerThread + i) % kNumKeys];
                    if (dist(local_rng) < kReadRatio) {
                        sink = mutex_sketch.count(k);
                    } else {
                        mutex_sketch.increment(k);
                    }
                }
                (void)sink;
            });
        }

        start.store(true, std::memory_order_release);
        for (auto &th : threads) th.join();

        double ms = global_timer.elapsedMs();
        size_t total_ops = kNumThreads * kOpsPerThread;
        std::cout << "[bench] Mutex   mixed(" << int(kReadRatio * 100)
                  << "% read) x " << total_ops << " (" << kNumThreads
                  << " threads) = " << ms << " ms  ("
                  << (total_ops / ms * 1000 / 1e6) << " Mops/s)" << std::endl;
    }

    // Benchmark for atomic sketch
    {
        std::atomic<bool> start{false};
        std::vector<std::thread> threads;
        std::mt19937 rng(42);

        Timer global_timer;
        for (size_t t = 0; t < kNumThreads; ++t) {
            threads.emplace_back([&, t]() {
                std::mt19937 local_rng(42 + t);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                while (!start.load(std::memory_order_acquire)) {
                }
                volatile uint8_t sink = 0;
                for (size_t i = 0; i < kOpsPerThread; ++i) {
                    const auto &k = keys[(t * kOpsPerThread + i) % kNumKeys];
                    if (dist(local_rng) < kReadRatio) {
                        sink = atomic_sketch.count(k);
                    } else {
                        atomic_sketch.increment(k);
                    }
                }
                (void)sink;
            });
        }

        start.store(true, std::memory_order_release);
        for (auto &th : threads) th.join();

        double ms = global_timer.elapsedMs();
        size_t total_ops = kNumThreads * kOpsPerThread;
        std::cout << "[bench] Atomic  mixed(" << int(kReadRatio * 100)
                  << "% read) x " << total_ops << " (" << kNumThreads
                  << " threads) = " << ms << " ms  ("
                  << (total_ops / ms * 1000 / 1e6) << " Mops/s)" << std::endl;
    }
}

}  // namespace
}  // namespace mooncake
