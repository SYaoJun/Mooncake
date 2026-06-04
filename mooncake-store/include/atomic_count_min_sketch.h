#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace mooncake {

// Lock-free Count-Min Sketch using std::atomic<uint8_t>.
// All read/write operations on the table are lock-free,
// but decay() still uses a mutex to ensure consistency.
class AtomicCountMinSketch {
   public:
    explicit AtomicCountMinSketch(size_t width = 4096, size_t depth = 4)
        : width_(width > 0 ? width : kDefaultWidth),
          depth_(depth > 0 ? depth : kDefaultDepth),
          table_(width_ * depth_),
          total_increments_(0) {}

    // Increment the count for |key| and return the estimated min-count.
    // Uses CAS loop for thread-safe increment.
    uint8_t increment(const std::string &key) {
        uint8_t min_val = UINT8_MAX;
        for (size_t i = 0; i < depth_; ++i) {
            size_t idx = hash(key, i) % width_;
            min_val = std::min(min_val, incrementCell(i * width_ + idx));
        }
        size_t prev = total_increments_.fetch_add(1, std::memory_order_relaxed);
        if (prev + 1 >= width_ * depth_) {
            decay();
        }
        return min_val;
    }

    // Return the estimated count for |key| (read-only, lock-free).
    uint8_t count(const std::string &key) const {
        uint8_t min_val = UINT8_MAX;
        for (size_t i = 0; i < depth_; ++i) {
            size_t idx = hash(key, i) % width_;
            uint8_t val = table_[i * width_ + idx].load(std::memory_order_relaxed);
            min_val = std::min(min_val, val);
        }
        return min_val;
    }

    // Halve all counters (right-shift by 1). Uses mutex for decay.
    void decay() {
        std::lock_guard<std::mutex> lock(decay_mu_);
        for (size_t i = 0; i < width_ * depth_; ++i) {
            table_[i].store(table_[i].load(std::memory_order_relaxed) >> 1,
                            std::memory_order_relaxed);
        }
        total_increments_.store(0, std::memory_order_relaxed);
    }

   private:
    static constexpr size_t kDefaultWidth = 4096;
    static constexpr size_t kDefaultDepth = 4;

    // Atomically increment a cell, saturating at UINT8_MAX.
    uint8_t incrementCell(size_t index) {
        auto &cell = table_[index];
        uint8_t expected = cell.load(std::memory_order_relaxed);
        while (expected < UINT8_MAX) {
            if (cell.compare_exchange_weak(expected, expected + 1,
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
                return expected + 1;
            }
            // expected was updated by CAS; retry
        }
        return UINT8_MAX;
    }

    size_t hash(const std::string &key, size_t seed) const {
        size_t h = std::hash<std::string>{}(key);
        h ^= seed * 0x9e3779b97f4a7c15ULL + 0x517cc1b727220a95ULL;
        h ^= (h >> 33);
        h *= 0xff51afd7ed558ccdULL;
        h ^= (h >> 33);
        return h;
    }

    const size_t width_;
    const size_t depth_;
    std::vector<std::atomic<uint8_t>> table_;
    std::atomic<size_t> total_increments_;
    std::mutex decay_mu_;
};

}  // namespace mooncake
