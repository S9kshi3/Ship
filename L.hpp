
#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <fstream>
#include <thread>
#include <memory>
#include <sstream>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <deque>
#include <map>
#include <unordered_map>
#include <cstdio>
#include <shared_mutex>
#include <atomic>
#include <source_location>
#include <format>
#include <string_view>
#include <type_traits>
#include <asio.hpp>
#include <queue>
#include <functional>
#include <future>
#include <mutex>
#include <condition_variable>
#include <utility>
#include <variant>
#include <span>
#include <array>
#include <cstring>
#include <filesystem>

// RapidJSON Includes
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

// ============================================================================
// PERFORMANCE MACROS & OS OPTIMIZATIONS
// ============================================================================
#if defined(_MSC_VER)
#define LSTREAM_FORCE_INLINE __forceinline
#include <io.h>
#define ISATTY _isatty
#define FILENO _fileno
#elif defined(__GNUC__) || defined(__clang__)
#define LSTREAM_FORCE_INLINE inline __attribute__((always_inline))
#include <unistd.h>
#define ISATTY isatty
#define FILENO fileno
#else
#define LSTREAM_FORCE_INLINE inline
#endif

constexpr size_t LSTREAM_CACHE_LINE = 64;

namespace Lstream {

    // Compile-Time Metadata Stripping
#ifdef LSTREAM_STRIP_SOURCE_LOCATION
    struct SourceLoc {
        static constexpr SourceLoc current() noexcept { return {}; }
        constexpr const char* file_name() const noexcept { return ""; }
        constexpr const char* function_name() const noexcept { return ""; }
        constexpr uint32_t line() const noexcept { return 0; }
        constexpr uint32_t column() const noexcept { return 0; }
    };
#else
    using SourceLoc = std::source_location;
#endif

    // Zero-Overhead Synchronization Primitives
    struct NullMutex {
        LSTREAM_FORCE_INLINE void lock() noexcept {}
        LSTREAM_FORCE_INLINE void unlock() noexcept {}
        LSTREAM_FORCE_INLINE bool try_lock() noexcept { return true; }
    };

    struct NullSharedMutex {
        LSTREAM_FORCE_INLINE void lock() noexcept {}
        LSTREAM_FORCE_INLINE void unlock() noexcept {}
        LSTREAM_FORCE_INLINE void lock_shared() noexcept {}
        LSTREAM_FORCE_INLINE void unlock_shared() noexcept {}
        LSTREAM_FORCE_INLINE bool try_lock() noexcept { return true; }
        LSTREAM_FORCE_INLINE bool try_lock_shared() noexcept { return true; }
    };

    // Native Fast File I/O
    class NativeFile {
        std::FILE* fp_{ nullptr };
    public:
        NativeFile() = default;
        ~NativeFile() { close(); }

        bool open(const std::string& path, bool append = true) {
            close();
#ifdef _WIN32
            fp_ = _fsopen(path.c_str(), append ? "ab" : "wb", _SH_DENYNO);
#else
            fp_ = std::fopen(path.c_str(), append ? "ab" : "wb");
#endif
            return fp_ != nullptr;
        }

        LSTREAM_FORCE_INLINE void write(const char* data, size_t size) {
            if (fp_) [[likely]] std::fwrite(data, 1, size, fp_);
        }

        LSTREAM_FORCE_INLINE void flush() { if (fp_) [[likely]] std::fflush(fp_); }
        void close() { if (fp_) { std::fclose(fp_); fp_ = nullptr; } }
    };

    // ============================================================================
    // MULTIDIMENSIONAL INDEX SYSTEM
    // ============================================================================
    struct MultiIndex {
        std::array<unsigned int, 16> values{};
        size_t count = 0;

        MultiIndex() = default;
        // FIX: Made explicit to prevent single integers from hijacking MultiIndex overloads
        explicit MultiIndex(unsigned int main_idx) : count(1) { values[0] = main_idx; }
        MultiIndex(std::initializer_list<unsigned int> init) {
            count = std::min(init.size(), values.size());
            std::copy_n(init.begin(), count, values.begin());
        }

        LSTREAM_FORCE_INLINE void Append(unsigned int val) {
            if (count < values.size()) [[likely]] values[count++] = val;
        }

        LSTREAM_FORCE_INLINE unsigned int MainIndex() const { return count > 0 ? values[0] : 0; }
        LSTREAM_FORCE_INLINE bool IsEmpty() const { return count == 0; }

        // Optimized ToString to prevent heavy allocations
        std::string ToString() const {
            if (count == 0) return "-";
            std::string res;
            res.reserve(count * 6);
            res += std::to_string(values[0]);
            for (size_t i = 1; i < count; ++i) {
                res += ':';
                res += std::to_string(values[i]);
            }
            return res;
        }
    };

    // ============================================================================
    // CATEGORICAL LOGGING DOMAINS
    // ============================================================================
    namespace Category {
        constexpr std::string_view Telemetry = "Telemetry";
        constexpr std::string_view Performance = "Performance";
        constexpr std::string_view Physics = "Physics";
        constexpr std::string_view Gameplay = "Gameplay";
        constexpr std::string_view Networking = "Networking";
        constexpr std::string_view Engine = "Engine";
    }
}

// ============================================================================
// THREAD POOL IMPLEMENTATIONS
// ============================================================================
namespace CBE {
    namespace UC {
        namespace UC_ThreadPool {
            namespace UC_Network {

                class SmartThreadPool {
                public:
                    explicit SmartThreadPool(size_t num_threads)
                        : io_context_(std::make_shared<asio::io_context>()),
                        work_guard_(asio::make_work_guard(*io_context_)),
                        num_threads_(num_threads),
                        is_running_(false) {
                    }

                    explicit SmartThreadPool(asio::io_context& io_context, size_t num_threads)
                        : io_context_(std::shared_ptr<asio::io_context>(&io_context, [](asio::io_context*) {})),
                        work_guard_(asio::make_work_guard(*io_context_)),
                        num_threads_(num_threads),
                        is_running_(false) {
                    }

                    explicit SmartThreadPool(std::shared_ptr<asio::io_context> shared_io_context, size_t num_threads)
                        : io_context_(std::move(shared_io_context)),
                        work_guard_(asio::make_work_guard(*io_context_)),
                        num_threads_(num_threads),
                        is_running_(false) {
                    }

                    SmartThreadPool(const SmartThreadPool&) = delete;
                    SmartThreadPool& operator=(const SmartThreadPool&) = delete;

                    ~SmartThreadPool() { stop(); }

                    void activate() {
                        if (!is_running_) {
                            for (size_t i = 0; i < num_threads_; ++i) {
                                threads_.emplace_back([this]() { io_context_->run(); });
                            }
                            is_running_ = true;
                        }
                    }

                    void stop() {
                        if (is_running_) {
                            work_guard_.reset();
                            for (auto& thread : threads_) {
                                if (thread.joinable()) thread.join();
                            }
                            is_running_ = false;
                        }
                    }

                    template <typename F>
                    void enqueue(F&& task) {
                        if (is_running_) [[likely]] asio::post(*io_context_, std::forward<F>(task));
                    }

                    [[nodiscard]] asio::io_context& get_io_context() const {
#ifndef LSTREAM_DISABLE_VALIDATION
                        if (!io_context_) [[unlikely]] throw std::runtime_error("io_context is not initialized");
#endif
                        return *io_context_;
                    }

                private:
                    std::shared_ptr<asio::io_context> io_context_;
                    asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
                    std::vector<std::thread> threads_;
                    size_t num_threads_;
                    bool is_running_ = false;
                };
            } // namespace UC_Network

            class MoveOnlyTask {
                struct Concept {
                    virtual ~Concept() = default;
                    virtual void call() = 0;
                };
                template <typename F>
                struct Model : Concept {
                    F f;
                    Model(F&& f) : f(std::move(f)) {}
                    void call() override { f(); }
                };
                std::unique_ptr<Concept> ptr;
            public:
                MoveOnlyTask() = default;
                template <typename F>
                MoveOnlyTask(F&& f) : ptr(std::make_unique<Model<std::decay_t<F>>>(std::forward<F>(f))) {}
                MoveOnlyTask(MoveOnlyTask&&) = default;
                MoveOnlyTask& operator=(MoveOnlyTask&&) = default;
                LSTREAM_FORCE_INLINE void operator()() { if (ptr) [[likely]] ptr->call(); }
            };

            class SmartThreadPool {
            public:
                explicit SmartThreadPool(size_t num_threads = std::thread::hardware_concurrency())
                    : stop_(false) {
                    if (num_threads == 0) num_threads = 1;

                    for (size_t i = 0; i < num_threads; ++i) {
                        workers_.emplace_back([this] {
                            while (true) {
                                MoveOnlyTask task;
                                {
                                    std::unique_lock<std::mutex> lock(this->queue_mutex_);
                                    this->condition_.wait(lock, [this] {
                                        return this->stop_ || !this->tasks_.empty();
                                        });

                                    if (this->stop_ && this->tasks_.empty()) [[unlikely]] return;
                                    task = std::move(this->tasks_.front());
                                    this->tasks_.pop();
                                }
                                try { task(); }
                                catch (...) {}
                            }
                            });
                    }
                }

                SmartThreadPool(const SmartThreadPool&) = delete;
                SmartThreadPool& operator=(const SmartThreadPool&) = delete;

                ~SmartThreadPool() {
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        stop_ = true;
                    }
                    condition_.notify_all();
                    for (std::thread& worker : workers_) {
                        if (worker.joinable()) worker.join();
                    }
                }

                template<class F, class... Args>
                auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>> {
                    using return_type = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
                    std::packaged_task<return_type()> task(
                        [func = std::forward<F>(f), ...args = std::forward<Args>(args)]() mutable -> return_type {
                            return std::invoke(std::move(func), std::move(args)...);
                        }
                    );
                    std::future<return_type> res = task.get_future();
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
#ifndef LSTREAM_DISABLE_VALIDATION
                        if (stop_) [[unlikely]] throw std::runtime_error("enqueue on stopped ThreadPool");
#endif
                        tasks_.emplace([t = std::move(task)]() mutable { t(); });
                    }
                    condition_.notify_one();
                    return res;
                }

            private:
                std::vector<std::thread> workers_;
                std::queue<MoveOnlyTask> tasks_;
                std::mutex queue_mutex_;
                std::condition_variable condition_;
                bool stop_;
            };
        } // namespace UC_ThreadPool
    } // namespace UC
} // namespace CBE

// ============================================================================
// LEGACY LOG CLASS (UPGRADED WITH MULTI-INDEX)
// ============================================================================
namespace Lstream {

    enum class LogMode { File, Cache, FileCache };

    template <class T>
    class Log {
    public:
        Log(int sz, LogMode mode = LogMode::Cache, const std::string& fname = "application_log.txt", int cache_sz = 10);
        ~Log();

        bool DoBoundCheck(unsigned int index) const;

        bool SetLog(const T& value, char status_char, bool auto_resize = true, const SourceLoc& loc = SourceLoc::current());
        bool SetLog(const T& value, const MultiIndex& idx, char status_char, const SourceLoc& loc = SourceLoc::current());

        bool IsOccupied(unsigned int index) const;

        T& RetriveLog(unsigned int index);
        const T& RetriveLog(unsigned int index) const;

        void ClearLog(unsigned int index);
        void Resize(int new_sz);

        unsigned int GetLastAccessedLogIndex() const;
        unsigned int GetCurrentLastOccupiedLogIndex() const;
        int GetSize() const;
        LogMode GetLogMode() const { return mode_; }

        void FlushFileLog();
        void SetIntervalReferenceTimestamp(std::chrono::high_resolution_clock::time_point new_timestamp);
        std::chrono::high_resolution_clock::time_point GetLastLogIntervalTimestamp() const;
        std::string GetFormattedLastLogIntervalTimestamp() const;

        std::vector<T> Svalue_;
        std::vector<char> Occupied_status_;

    private:
        alignas(LSTREAM_CACHE_LINE) mutable std::atomic<unsigned int> last_accessed_log_index_{ 0 };
        alignas(LSTREAM_CACHE_LINE) mutable std::atomic<unsigned int> current_last_occupied_log_index_{ 0 };
        alignas(LSTREAM_CACHE_LINE) mutable std::shared_mutex rw_mutex_;

        std::chrono::high_resolution_clock::time_point last_log_timestamp_;

        int size_;
        LogMode mode_;
        std::string filename_;
        mutable std::fstream log_file_;

        std::deque<std::pair<unsigned int, T>> file_cache_queue_;
        int file_cache_size_;
        std::map<unsigned int, long long> file_index_offsets_;

        void UpdateLastLogIndex(unsigned int index) const;
        std::string GetTimeL() const;
        std::string GetTimeL(std::chrono::system_clock::time_point tp) const;
        std::string GetFormattedTimeInterval();

        template<typename U>
        std::string FormatLogLine(const MultiIndex& idx, const U& message_part, const std::string& time_str, const std::string& interval_str, char status_char, const SourceLoc* loc = nullptr);

        void WriteLogEntryToFile(const MultiIndex& idx, const std::string& message_to_write, const std::string& time_str, const std::string& interval_str, char status_char, const SourceLoc* loc = nullptr);

        T ReadLogEntryFromFile(unsigned int main_index) const;
        void AddToCache(unsigned int index, const T& value);
        bool GetFromCache(unsigned int index, T& out_value) const;
        bool GetFromCacheMutable(unsigned int index, T& out_value);
    };

    // ------------------------------------------------------------------------
    // LEGACY LOG IMPLEMENTATIONS
    // ------------------------------------------------------------------------
    template <class T>
    Log<T>::Log(int sz, LogMode mode, const std::string& fname, int cache_sz)
        : size_(sz), mode_(mode), filename_(fname), file_cache_size_(cache_sz) {
        last_log_timestamp_ = std::chrono::high_resolution_clock::now();
        if (mode_ == LogMode::Cache || mode_ == LogMode::FileCache) {
            Svalue_.resize(size_);
            Occupied_status_.resize(size_, 0);
        }
    }

    template <class T>
    Log<T>::~Log() {
        FlushFileLog();
    }

    template <class T>
    bool Log<T>::DoBoundCheck(unsigned int index) const {
        return index < static_cast<unsigned int>(size_);
    }

    template <class T>
    void Log<T>::UpdateLastLogIndex(unsigned int index) const {
        last_accessed_log_index_.store(index, std::memory_order_relaxed);
    }

    template <class T>
    std::string Log<T>::GetTimeL() const {
        return GetTimeL(std::chrono::system_clock::now());
    }

    template <class T>
    std::string Log<T>::GetTimeL(std::chrono::system_clock::time_point tp) const {
        std::time_t now_c = std::chrono::system_clock::to_time_t(tp);
        std::tm now_tm;
#ifdef _WIN32
        localtime_s(&now_tm, &now_c);
#else
        localtime_r(&now_c, &now_tm);
#endif
        std::stringstream ss;
        ss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    template <class T>
    std::string Log<T>::GetFormattedTimeInterval() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_timestamp_);
        last_log_timestamp_ = now;
        return std::to_string(duration.count()) + "ms";
    }

    template <class T>
    template <typename U>
    std::string Log<T>::FormatLogLine(const MultiIndex& idx, const U& message_part, const std::string& time_str, const std::string& interval_str, char status_char, const SourceLoc* loc) {
        std::string prefix;
        if (loc) {
            std::string_view file = loc->file_name();
            auto last_slash = file.find_last_of("/\\");
            if (last_slash != std::string_view::npos) file = file.substr(last_slash + 1);
            prefix = std::format("{} [{}] - [{}] ->[{}] [{}:{}] : ", idx.ToString(), time_str, interval_str, status_char, file, loc->line());
        }
        else {
            prefix = std::format("{} [{}] - [{}] ->[{}] : ", idx.ToString(), time_str, interval_str, status_char);
        }

        if constexpr (std::is_same_v<std::decay_t<U>, std::string>) {
            return prefix + message_part;
        }
        else if constexpr (std::is_convertible_v<U, std::string_view>) {
            return prefix + std::string(std::string_view(message_part));
        }
        else {
            std::stringstream ss;
            ss << prefix << message_part;
            return ss.str();
        }
    }

    template <class T>
    void Log<T>::WriteLogEntryToFile(const MultiIndex& idx, const std::string& message_to_write, const std::string& time_str, const std::string& interval_str, char status_char, const SourceLoc* loc) {
        if (mode_ == LogMode::File || mode_ == LogMode::FileCache) {
            if (!log_file_.is_open()) {
                log_file_.open(filename_, std::ios::out | std::ios::app);
                if (!log_file_.is_open()) throw std::runtime_error("Failed to open log file for writing: " + filename_);
            }
            if (idx.count <= 1 || (idx.count > 1 && idx.values[idx.count - 1] == 0)) {
                file_index_offsets_[idx.MainIndex()] = log_file_.tellp();
            }
            log_file_ << FormatLogLine(idx, message_to_write, time_str, interval_str, status_char, loc) << '\n';
        }
    }

    template <class T>
    bool Log<T>::SetLog(const T& value, const MultiIndex& idx, char status_char, const SourceLoc& loc) {
        unsigned int main_index = idx.MainIndex();
        UpdateLastLogIndex(main_index);

        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        std::string time_str = GetTimeL();
        std::string interval_str = GetFormattedTimeInterval();

        if (mode_ == LogMode::File || mode_ == LogMode::FileCache) {
            if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                unsigned int sub_index = 0;
                for (const auto& msg : value) {
                    MultiIndex sub_idx = idx;
                    sub_idx.Append(sub_index++);
                    WriteLogEntryToFile(sub_idx, msg, time_str, interval_str, status_char, &loc);
                }
            }
            else {
                std::stringstream ss;
                ss << value;
                WriteLogEntryToFile(idx, ss.str(), time_str, interval_str, status_char, &loc);
            }
        }

        if (mode_ == LogMode::Cache || mode_ == LogMode::FileCache) {
            if (!DoBoundCheck(main_index)) {
                throw std::out_of_range(std::format("[SetLog Function] Input index is out of bounds for memory allocation: {}", main_index));
            }
            Svalue_[main_index] = value;
            Occupied_status_[main_index] = 1;

            unsigned int current_last = current_last_occupied_log_index_.load(std::memory_order_relaxed);
            while (main_index > current_last && !current_last_occupied_log_index_.compare_exchange_weak(current_last, main_index, std::memory_order_relaxed)) {}
        }

        if (mode_ == LogMode::FileCache) {
            AddToCache(main_index, value);
        }
        return true;
    }

    template <class T>
    bool Log<T>::SetLog(const T& value, char status_char, bool auto_resize, const SourceLoc& loc) {
        unsigned int next_index;
        {
            std::unique_lock<std::shared_mutex> lock(rw_mutex_);

            if ((mode_ == LogMode::Cache || mode_ == LogMode::FileCache) && (size_ == 0 || (current_last_occupied_log_index_.load(std::memory_order_relaxed) == 0 && Occupied_status_[0] == 0))) {
                next_index = 0;
            }
            else if (mode_ == LogMode::File && current_last_occupied_log_index_.load(std::memory_order_relaxed) == 0 && file_index_offsets_.empty()) {
                next_index = 0;
            }
            else {
                next_index = current_last_occupied_log_index_.load(std::memory_order_relaxed) + 1;
            }

            if ((mode_ == LogMode::Cache || mode_ == LogMode::FileCache) && !DoBoundCheck(next_index)) {
                if (auto_resize) {
                    int new_sz = std::max(size_ + 1, size_ * 2);
                    size_ = new_sz;
                    Svalue_.resize(new_sz);
                    Occupied_status_.resize(new_sz, 0);
                }
                else {
                    throw std::out_of_range("[SetLog Auto] Log is full and auto_resize is false. Cannot add new entry.");
                }
            }

            current_last_occupied_log_index_.store(next_index, std::memory_order_relaxed);
        }

        return SetLog(value, MultiIndex{ next_index }, status_char, loc);
    }

    template <class T>
    bool Log<T>::IsOccupied(unsigned int index) const {
        if (mode_ == LogMode::File) return file_index_offsets_.find(index) != file_index_offsets_.end();
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        return DoBoundCheck(index) && Occupied_status_[index] == 1;
    }

    template <class T>
    T& Log<T>::RetriveLog(unsigned int index) {
        UpdateLastLogIndex(index);
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        if (mode_ == LogMode::File) throw std::logic_error("RetriveLog(mutable) not supported in File mode.");
        if (mode_ == LogMode::FileCache) {
            T cached_val;
            if (GetFromCacheMutable(index, cached_val)) return Svalue_[index];
            Svalue_[index] = ReadLogEntryFromFile(index);
            AddToCache(index, Svalue_[index]);
            return Svalue_[index];
        }
        if (!DoBoundCheck(index)) throw std::out_of_range("Index out of bounds");
        return Svalue_[index];
    }

    template <class T>
    const T& Log<T>::RetriveLog(unsigned int index) const {
        UpdateLastLogIndex(index);
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        if (mode_ == LogMode::File) throw std::logic_error("RetriveLog(const) not supported in File mode. Use ReadLogEntryFromFile.");
        if (mode_ == LogMode::FileCache) {
            T cached_val;
            if (GetFromCache(index, cached_val)) {
                static thread_local T temp_val;
                temp_val = cached_val;
                return temp_val;
            }
            static thread_local T file_val;
            file_val = ReadLogEntryFromFile(index);
            return file_val;
        }
        if (!DoBoundCheck(index)) throw std::out_of_range("Index out of bounds");
        return Svalue_[index];
    }

    template <class T>
    void Log<T>::ClearLog(unsigned int index) {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        if (mode_ == LogMode::Cache || mode_ == LogMode::FileCache) {
            if (DoBoundCheck(index)) {
                Svalue_[index] = T();
                Occupied_status_[index] = 0;
            }
        }
        if (mode_ == LogMode::FileCache) {
            auto it = std::find_if(file_cache_queue_.begin(), file_cache_queue_.end(),
                [index](const std::pair<unsigned int, T>& p) { return p.first == index; });
            if (it != file_cache_queue_.end()) file_cache_queue_.erase(it);
        }
        if (mode_ == LogMode::File || mode_ == LogMode::FileCache) file_index_offsets_.erase(index);
    }

    template <class T>
    void Log<T>::Resize(int new_sz) {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        if (mode_ == LogMode::Cache || mode_ == LogMode::FileCache) {
            Svalue_.resize(new_sz);
            Occupied_status_.resize(new_sz, 0);
        }
        size_ = new_sz;
    }

    template <class T>
    unsigned int Log<T>::GetLastAccessedLogIndex() const { return last_accessed_log_index_.load(std::memory_order_relaxed); }

    template <class T>
    unsigned int Log<T>::GetCurrentLastOccupiedLogIndex() const { return current_last_occupied_log_index_.load(std::memory_order_relaxed); }

    template <class T>
    int Log<T>::GetSize() const { return size_; }

    template <class T>
    void Log<T>::FlushFileLog() {
        if (log_file_.is_open()) log_file_.flush();
    }

    template <class T>
    void Log<T>::SetIntervalReferenceTimestamp(std::chrono::high_resolution_clock::time_point new_timestamp) { last_log_timestamp_ = new_timestamp; }

    template <class T>
    std::chrono::high_resolution_clock::time_point Log<T>::GetLastLogIntervalTimestamp() const { return last_log_timestamp_; }

    template <class T>
    std::string Log<T>::GetFormattedLastLogIntervalTimestamp() const { return GetTimeL(std::chrono::system_clock::now()); }

    template <class T>
    T Log<T>::ReadLogEntryFromFile(unsigned int main_index) const {
        if (file_index_offsets_.find(main_index) == file_index_offsets_.end()) throw std::out_of_range("Index not found in file.");
        long long offset = file_index_offsets_.at(main_index);
        std::fstream read_file(filename_, std::ios::in);
        if (!read_file.is_open()) throw std::runtime_error("Could not open log file for reading.");
        read_file.seekg(offset);
        std::string line;
        std::getline(read_file, line);
        if constexpr (std::is_same_v<T, std::string>) return line;
        else {
            T val;
            std::stringstream ss(line);
            ss >> val;
            return val;
        }
    }

    template <class T>
    void Log<T>::AddToCache(unsigned int index, const T& value) {
        auto it = std::find_if(file_cache_queue_.begin(), file_cache_queue_.end(),
            [index](const std::pair<unsigned int, T>& p) { return p.first == index; });
        if (it != file_cache_queue_.end()) file_cache_queue_.erase(it);
        if (file_cache_queue_.size() >= static_cast<size_t>(file_cache_size_)) file_cache_queue_.pop_front();
        file_cache_queue_.push_back({ index, value });
    }

    template <class T>
    bool Log<T>::GetFromCache(unsigned int index, T& out_value) const {
        auto it = std::find_if(file_cache_queue_.begin(), file_cache_queue_.end(),
            [index](const std::pair<unsigned int, T>& p) { return p.first == index; });
        if (it != file_cache_queue_.end()) {
            out_value = it->second;
            return true;
        }
        return false;
    }

    template <class T>
    bool Log<T>::GetFromCacheMutable(unsigned int index, T& out_value) {
        auto it = std::find_if(file_cache_queue_.begin(), file_cache_queue_.end(),
            [index](const std::pair<unsigned int, T>& p) { return p.first == index; });
        if (it != file_cache_queue_.end()) {
            out_value = it->second;
            auto val = *it;
            file_cache_queue_.erase(it);
            file_cache_queue_.push_back(val);
            return true;
        }
        return false;
    }
}

// ============================================================================
// NEW ARCHITECTURE: FRONT-END, PIPELINE, AND BACK-END
// ============================================================================
namespace Lstream {

    enum class Severity { Trace, Debug, Info, Warning, Error, Critical };

    constexpr std::string_view ToString(Severity sev) {
        switch (sev) {
        case Severity::Trace:    return "TRACE";
        case Severity::Debug:    return "DEBUG";
        case Severity::Info:     return "INFO";
        case Severity::Warning:  return "WARN";
        case Severity::Error:    return "ERROR";
        case Severity::Critical: return "FATAL";
        }
        return "UNKNOWN";
    }

    constexpr std::string_view GetAnsiColor(Severity sev) {
        switch (sev) {
        case Severity::Trace:    return "\033[37m";
        case Severity::Debug:    return "\033[36m";
        case Severity::Info:     return "\033[32m";
        case Severity::Warning:  return "\033[33m";
        case Severity::Error:    return "\033[31m";
        case Severity::Critical: return "\033[1;31m";
        }
        return "";
    }

    Severity ParseSeverity(std::string_view str) {
        if (str == "TRACE") return Severity::Trace;
        if (str == "DEBUG") return Severity::Debug;
        if (str == "INFO") return Severity::Info;
        if (str == "WARN" || str == "WARNING") return Severity::Warning;
        if (str == "ERROR") return Severity::Error;
        if (str == "FATAL" || str == "CRITICAL") return Severity::Critical;
        return Severity::Info;
    }

    using AttributeValue = std::variant<int, double, std::string, const void*>;
    struct Attribute {
        std::string key;
        AttributeValue value;
    };

    struct LogMetadata {
        Severity severity;
        SourceLoc source_loc;
        std::chrono::system_clock::time_point timestamp;
        std::thread::id thread_id;
        std::string_view logger_namespace;
        MultiIndex index;
    };

    struct LogRecord {
        LogMetadata metadata;
        std::string_view formatted_message;
        std::span<const Attribute> attributes;
        std::span<const std::byte> binary_payload;

        std::array<Attribute, 16> injected_attributes{};
        size_t injected_count = 0;

        std::array<std::byte, 1024> stack_binary_buf{};

        std::string owned_message;
        std::vector<Attribute> owned_attributes;
        std::vector<std::byte> owned_payload;

        LogRecord() = default;

        LogRecord(const LogRecord& other)
            : metadata(other.metadata),
            injected_attributes(other.injected_attributes),
            injected_count(other.injected_count),
            stack_binary_buf(other.stack_binary_buf),
            owned_message(other.owned_message),
            owned_attributes(other.owned_attributes),
            owned_payload(other.owned_payload)
        {
            formatted_message = !owned_message.empty() ? owned_message : other.formatted_message;
            attributes = !owned_attributes.empty() ? owned_attributes : other.attributes;
            binary_payload = !owned_payload.empty() ? owned_payload : other.binary_payload;
        }

        LogRecord(LogRecord&& other) noexcept
            : metadata(std::move(other.metadata)),
            injected_attributes(std::move(other.injected_attributes)),
            injected_count(other.injected_count),
            stack_binary_buf(std::move(other.stack_binary_buf)),
            owned_message(std::move(other.owned_message)),
            owned_attributes(std::move(other.owned_attributes)),
            owned_payload(std::move(other.owned_payload))
        {
            formatted_message = !owned_message.empty() ? owned_message : other.formatted_message;
            attributes = !owned_attributes.empty() ? owned_attributes : other.attributes;
            binary_payload = !owned_payload.empty() ? owned_payload : other.binary_payload;

            other.formatted_message = {};
            other.attributes = {};
            other.binary_payload = {};
            other.injected_count = 0;
        }

        LogRecord& operator=(const LogRecord&) = delete;
        LogRecord& operator=(LogRecord&&) = delete;
    };

    class ISink {
    public:
        virtual ~ISink() = default;
        virtual void Log(const LogRecord& record) = 0;
        virtual void Flush() = 0;
    };

    template <typename F> struct LazyEval { F func; };
    template <typename F> LazyEval(F) -> LazyEval<F>;

    // FIX: Constrained LogFormatString constructor to prevent ambiguity with MultiIndex overloads
    template <typename... Args>
    struct LogFormatString {
        std::format_string<Args...> fmt;
        SourceLoc loc;

        template <typename T, typename = std::enable_if_t<std::is_convertible_v<const T&, std::string_view>>>
        consteval LogFormatString(const T& s, SourceLoc l = SourceLoc::current())
            : fmt(s), loc(l) {
        }
    };

    // ============================================================================
    // SERIALIZATION ENGINE
    // ============================================================================
    template <typename T, typename Enable = void>
    struct Serializer {};

    template <typename T>
    struct Serializer<T, std::enable_if_t<std::is_trivial_v<T>&& std::is_standard_layout_v<T>>> {
        static size_t Measure(const T&) { return sizeof(T); }
        static void Serialize(const T& obj, std::byte* out) { std::memcpy(out, &obj, sizeof(T)); }
    };

    template <>
    struct Serializer<std::span<const std::byte>> {
        static size_t Measure(std::span<const std::byte> obj) { return sizeof(uint32_t) + obj.size(); }
        static void Serialize(std::span<const std::byte> obj, std::byte* out) {
            uint32_t len = static_cast<uint32_t>(obj.size());
            std::memcpy(out, &len, sizeof(len));
            std::memcpy(out + sizeof(len), obj.data(), obj.size());
        }
    };

    template <>
    struct Serializer<std::span<std::byte>> {
        static size_t Measure(std::span<std::byte> obj) { return sizeof(uint32_t) + obj.size(); }
        static void Serialize(std::span<std::byte> obj, std::byte* out) {
            uint32_t len = static_cast<uint32_t>(obj.size());
            std::memcpy(out, &len, sizeof(len));
            std::memcpy(out + sizeof(len), obj.data(), obj.size());
        }
    };

    template <>
    struct Serializer<std::vector<std::byte>> {
        static size_t Measure(const std::vector<std::byte>& obj) { return sizeof(uint32_t) + obj.size(); }
        static void Serialize(const std::vector<std::byte>& obj, std::byte* out) {
            uint32_t len = static_cast<uint32_t>(obj.size());
            std::memcpy(out, &len, sizeof(len));
            std::memcpy(out + sizeof(len), obj.data(), obj.size());
        }
    };

    template <>
    struct Serializer<std::string> {
        static size_t Measure(const std::string& obj) { return sizeof(uint32_t) + obj.size(); }
        static void Serialize(const std::string& obj, std::byte* out) {
            uint32_t len = static_cast<uint32_t>(obj.size());
            std::memcpy(out, &len, sizeof(len));
            std::memcpy(out + sizeof(len), obj.data(), obj.size());
        }
    };

    template <>
    struct Serializer<std::string_view> {
        static size_t Measure(std::string_view obj) { return sizeof(uint32_t) + obj.size(); }
        static void Serialize(std::string_view obj, std::byte* out) {
            uint32_t len = static_cast<uint32_t>(obj.size());
            std::memcpy(out, &len, sizeof(len));
            std::memcpy(out + sizeof(len), obj.data(), obj.size());
        }
    };

    class BinarySerializerWriter {
        std::byte* ptr_;
    public:
        explicit BinarySerializerWriter(std::byte* ptr) : ptr_(ptr) {}
        template <typename T>
        void Write(const T& value) {
            size_t size = Serializer<T>::Measure(value);
            Serializer<T>::Serialize(value, ptr_);
            ptr_ += size;
        }
        void WriteRaw(const void* data, size_t size) {
            std::memcpy(ptr_, data, size);
            ptr_ += size;
        }
    };

    // ============================================================================
    // FORMATTING ENGINES
    // ============================================================================
    class IFormatter {
    public:
        virtual ~IFormatter() = default;
        virtual void Format(LogRecord& record) const = 0;
    };

    class PlainTextFormatter : public IFormatter {
    public:
        void Format(LogRecord& record) const override {
            record.owned_message = std::format("[{:%Y-%m-%d %H:%M:%S}] [{}] [{}] [{}] {}",
                record.metadata.timestamp, record.metadata.logger_namespace,
                record.metadata.index.ToString(), ToString(record.metadata.severity),
                record.formatted_message);
            record.formatted_message = record.owned_message;
        }
    };

    class PatternFormatter : public IFormatter {
    public:
        using Resolver = std::function<std::string(const LogRecord&)>;
    private:
        enum class TokenType { Literal, Custom, Timestamp, Level, Logger, Index, Message, Thread, Color, Reset };
        struct Token { TokenType type; std::string literal; Resolver resolver; };

        std::vector<Token> tokens_;
        std::string timestamp_format_ = "{:%Y-%m-%d %H:%M:%S}";
        std::unordered_map<std::string, Resolver> custom_resolvers_;

        void ParsePattern(std::string_view pattern) {
            tokens_.clear();
            size_t pos = 0;
            while (pos < pattern.size()) {
                size_t start = pattern.find('{', pos);
                if (start == std::string_view::npos) {
                    tokens_.push_back({ TokenType::Literal, std::string(pattern.substr(pos)), nullptr });
                    break;
                }
                if (start + 1 < pattern.size() && pattern[start + 1] == '{') {
                    if (start > pos) tokens_.push_back({ TokenType::Literal, std::string(pattern.substr(pos, start - pos)), nullptr });
                    tokens_.push_back({ TokenType::Literal, "{", nullptr });
                    pos = start + 2;
                    continue;
                }

                if (start > pos) tokens_.push_back({ TokenType::Literal, std::string(pattern.substr(pos, start - pos)), nullptr });
                size_t end = pattern.find('}', start);
                if (end == std::string_view::npos) {
                    tokens_.push_back({ TokenType::Literal, std::string(pattern.substr(start)), nullptr });
                    break;
                }

                std::string key = std::string(pattern.substr(start + 1, end - start - 1));

                if (key == "timestamp") tokens_.push_back({ TokenType::Timestamp, "", nullptr });
                else if (key == "level") tokens_.push_back({ TokenType::Level, "", nullptr });
                else if (key == "logger") tokens_.push_back({ TokenType::Logger, "", nullptr });
                else if (key == "index") tokens_.push_back({ TokenType::Index, "", nullptr });
                else if (key == "message") tokens_.push_back({ TokenType::Message, "", nullptr });
                else if (key == "thread") tokens_.push_back({ TokenType::Thread, "", nullptr });
                else if (key == "color") tokens_.push_back({ TokenType::Color, "", nullptr });
                else if (key == "reset") tokens_.push_back({ TokenType::Reset, "", nullptr });
                else if (custom_resolvers_.count(key)) tokens_.push_back({ TokenType::Custom, "", custom_resolvers_[key] });
                else tokens_.push_back({ TokenType::Literal, std::string(pattern.substr(start, end - start + 1)), nullptr });

                pos = end + 1;
            }
        }

    public:
        explicit PatternFormatter(std::string_view pattern) { ParsePattern(pattern); }
        void SetTimestampFormat(std::string_view fmt) { timestamp_format_ = fmt; }
        void AddPlaceholder(std::string_view name, Resolver resolver) { custom_resolvers_[std::string(name)] = std::move(resolver); }

        void Format(LogRecord& record) const override {
            std::string out;
            for (const auto& t : tokens_) {
                switch (t.type) {
                case TokenType::Literal: out += t.literal; break;
                case TokenType::Timestamp: out += std::vformat(timestamp_format_, std::make_format_args(record.metadata.timestamp)); break;
                case TokenType::Level: out += ToString(record.metadata.severity); break;
                case TokenType::Logger: out += record.metadata.logger_namespace; break;
                case TokenType::Index: out += record.metadata.index.ToString(); break;
                case TokenType::Message: out += record.formatted_message; break;
                case TokenType::Color: out += GetAnsiColor(record.metadata.severity); break;
                case TokenType::Reset: out += "\033[0m"; break;
                case TokenType::Thread: {
                    std::stringstream ss; ss << record.metadata.thread_id;
                    out += ss.str();
                    break;
                }
                case TokenType::Custom: out += t.resolver(record); break;
                }
            }
            record.owned_message = std::move(out);
            record.formatted_message = record.owned_message;
        }
    };

    class JsonFormatter : public IFormatter {
    public:
        void Format(LogRecord& record) const override {
            rapidjson::StringBuffer sb;
            rapidjson::Writer<rapidjson::StringBuffer> writer(sb);

            writer.StartObject();
            writer.Key("timestamp");
            std::string ts = std::format("{:%Y-%m-%dT%H:%M:%S}", record.metadata.timestamp);
            writer.String(ts.c_str(), static_cast<rapidjson::SizeType>(ts.length()));

            writer.Key("level");
            std::string_view lvl = ToString(record.metadata.severity);
            writer.String(lvl.data(), static_cast<rapidjson::SizeType>(lvl.length()));

            writer.Key("logger");
            writer.String(record.metadata.logger_namespace.data(), static_cast<rapidjson::SizeType>(record.metadata.logger_namespace.length()));

            if (!record.metadata.index.IsEmpty()) {
                writer.Key("index");
                writer.StartArray();
                for (size_t i = 0; i < record.metadata.index.count; ++i) {
                    writer.Uint(record.metadata.index.values[i]);
                }
                writer.EndArray();
            }

            writer.Key("message");
            writer.String(record.formatted_message.data(), static_cast<rapidjson::SizeType>(record.formatted_message.length()));

            if (!record.attributes.empty()) {
                writer.Key("attributes");
                writer.StartObject();
                for (const auto& attr : record.attributes) {
                    writer.Key(attr.key.data(), static_cast<rapidjson::SizeType>(attr.key.length()));
                    std::visit([&writer](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, int>) writer.Int(arg);
                        else if constexpr (std::is_same_v<T, double>) writer.Double(arg);
                        else if constexpr (std::is_same_v<T, std::string>) writer.String(arg.c_str(), static_cast<rapidjson::SizeType>(arg.length()));
                        else {
                            std::stringstream ss; ss << arg;
                            std::string s = ss.str();
                            writer.String(s.c_str(), static_cast<rapidjson::SizeType>(s.length()));
                        }
                        }, attr.value);
                }
                writer.EndObject();
            }
            writer.EndObject();

            record.owned_message = sb.GetString();
            record.formatted_message = record.owned_message;
        }
    };

    class BinaryFormatter : public IFormatter {
    public:
        void Format(LogRecord& record) const override {
            std::vector<std::byte> payload;
            auto ts = record.metadata.timestamp.time_since_epoch().count();
            auto sev = static_cast<uint8_t>(record.metadata.severity);
            uint32_t msg_len = static_cast<uint32_t>(record.formatted_message.size());

            payload.resize(sizeof(ts) + sizeof(sev) + sizeof(msg_len) + msg_len);

            size_t offset = 0;
            std::memcpy(payload.data() + offset, &ts, sizeof(ts)); offset += sizeof(ts);
            std::memcpy(payload.data() + offset, &sev, sizeof(sev)); offset += sizeof(sev);
            std::memcpy(payload.data() + offset, &msg_len, sizeof(msg_len)); offset += sizeof(msg_len);
            std::memcpy(payload.data() + offset, record.formatted_message.data(), msg_len);

            record.owned_payload = std::move(payload);
            record.binary_payload = record.owned_payload;
        }
    };

    // ============================================================================
    // BACK-END: SINK INTERFACES & IMPLEMENTATIONS
    // ============================================================================
    template <typename Mutex = std::mutex>
    class ConsoleSinkImpl : public ISink {
        Mutex mutex_;
    public:
        void Log(const LogRecord& record) override {
            std::lock_guard<Mutex> lock(mutex_);
            std::cout << record.formatted_message << '\n';
        }
        void Flush() override { std::cout.flush(); }
    };

    template <typename Mutex = std::mutex>
    class RotatingFileSinkImpl : public ISink {
        std::string name_pattern_;
        size_t max_size_;
        size_t max_files_;
        size_t current_size_ = 0;
        NativeFile file_;
        Mutex mutex_;

        std::string GetFilename(size_t index) const { return std::vformat(name_pattern_, std::make_format_args(index)); }

        void Rotate() {
            file_.close();
            std::error_code ec;
            if (max_files_ > 0) {
                std::filesystem::remove(GetFilename(max_files_), ec);
                for (size_t i = max_files_; i > 0; --i) {
                    if (std::filesystem::exists(GetFilename(i - 1), ec)) {
                        std::filesystem::rename(GetFilename(i - 1), GetFilename(i), ec);
                    }
                }
            }
            file_.open(GetFilename(0), true);
            current_size_ = 0;
        }

    public:
        RotatingFileSinkImpl(std::string name_pattern, size_t max_size, size_t max_files)
            : name_pattern_(std::move(name_pattern)), max_size_(max_size), max_files_(max_files) {

            std::error_code ec;
            current_size_ = std::filesystem::file_size(GetFilename(0), ec);
            if (ec) current_size_ = 0;
            file_.open(GetFilename(0), true);
        }

        void Log(const LogRecord& record) override {
            std::lock_guard<Mutex> lock(mutex_);
            if (current_size_ + record.formatted_message.size() > max_size_) [[unlikely]] Rotate();
            file_.write(record.formatted_message.data(), record.formatted_message.size());
            file_.write("\n", 1);
            current_size_ += record.formatted_message.size() + 1;
        }
        void Flush() override {
            std::lock_guard<Mutex> lock(mutex_);
            file_.flush();
        }
    };

    class ICompressor {
    public:
        virtual ~ICompressor() = default;
        virtual std::vector<std::byte> Compress(std::span<const std::byte> data) = 0;
    };

    class NullCompressor : public ICompressor {
    public:
        std::vector<std::byte> Compress(std::span<const std::byte> data) override {
            return std::vector<std::byte>(data.begin(), data.end());
        }
    };

    template <typename Mutex = std::mutex>
    class BinarySinkImpl : public ISink {
        NativeFile file_;
        bool write_metadata_;
        std::shared_ptr<ICompressor> compressor_;
        Mutex mutex_;
    public:
        explicit BinarySinkImpl(const std::string& filename, bool write_metadata = true, std::shared_ptr<ICompressor> compressor = std::make_shared<NullCompressor>())
            : write_metadata_(write_metadata), compressor_(std::move(compressor)) {
            file_.open(filename, true);
        }

        void Log(const LogRecord& record) override {
            if (record.binary_payload.empty()) [[unlikely]] return;

            auto compressed = compressor_->Compress(record.binary_payload);

            std::lock_guard<Mutex> lock(mutex_);

            if (write_metadata_) {
                auto ts = record.metadata.timestamp.time_since_epoch().count();
                file_.write(reinterpret_cast<const char*>(&ts), sizeof(ts));

                uint32_t ns_size = static_cast<uint32_t>(record.metadata.logger_namespace.size());
                file_.write(reinterpret_cast<const char*>(&ns_size), sizeof(ns_size));
                file_.write(record.metadata.logger_namespace.data(), ns_size);

                uint32_t payload_size = static_cast<uint32_t>(compressed.size());
                file_.write(reinterpret_cast<const char*>(&payload_size), sizeof(payload_size));
            }

            file_.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
        }
        void Flush() override {
            std::lock_guard<Mutex> lock(mutex_);
            file_.flush();
        }
    };

    // ============================================================================
    // ASYNC QUEUE: DYNAMIC LOCK-FREE MPMC RING BUFFER
    // ============================================================================
    enum class OverflowPolicy { Block, Drop };

    class AsyncQueue : public std::enable_shared_from_this<AsyncQueue> {
        struct alignas(LSTREAM_CACHE_LINE) Slot {
            std::atomic<size_t> sequence;
            LogMetadata metadata;
            char* message_buf;
            size_t message_len;
            Attribute* attributes;
            size_t attr_count;
            std::byte* binary_buf;
            size_t binary_len;
        };

        std::vector<Slot> buffer_;
        size_t capacity_mask_;
        size_t max_msg_size_;
        size_t max_attrs_;
        size_t max_bin_size_;

        std::unique_ptr<char[]> message_pool_;
        std::unique_ptr<Attribute[]> attr_pool_;
        std::unique_ptr<std::byte[]> binary_pool_;

        alignas(LSTREAM_CACHE_LINE) std::atomic<size_t> enqueue_pos_{ 0 };
        alignas(LSTREAM_CACHE_LINE) std::atomic<size_t> dequeue_pos_{ 0 };
        alignas(LSTREAM_CACHE_LINE) std::atomic<size_t> active_tasks_{ 0 };
        alignas(LSTREAM_CACHE_LINE) std::atomic<size_t> dropped_count_{ 0 };

        CBE::UC::UC_ThreadPool::SmartThreadPool* pool_;
        OverflowPolicy policy_;
        std::function<void(LogRecord&)> processor_;

        void ScheduleConsumer() {
            size_t active = active_tasks_.load(std::memory_order_relaxed);
            if (active < 2) [[likely]] {
                if (active_tasks_.compare_exchange_strong(active, active + 1, std::memory_order_relaxed)) {
                    auto self = shared_from_this();
                    try {
                        pool_->enqueue([self]() { self->ProcessBatch(); });
                    }
                    catch (...) {
                        active_tasks_.fetch_sub(1, std::memory_order_relaxed);
                        ProcessBatch();
                    }
                }
            }
        }

        void ProcessBatch() {
            constexpr size_t BatchSize = 128;
            size_t processed = 0;

            while (processed < BatchSize) {
                size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
                Slot* slot = &buffer_[pos & capacity_mask_];
                size_t seq = slot->sequence.load(std::memory_order_acquire);
                intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);

                if (dif == 0) [[likely]] {
                    if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                        LogRecord rec;
                        rec.metadata = slot->metadata;
                        rec.formatted_message = std::string_view(slot->message_buf, slot->message_len);
                        rec.attributes = std::span<const Attribute>(slot->attributes, slot->attr_count);
                        rec.binary_payload = std::span<const std::byte>(slot->binary_buf, slot->binary_len);

                        processor_(rec);

                        slot->sequence.store(pos + capacity_mask_ + 1, std::memory_order_release);
                        processed++;
                    }
                }
                else break;
            }

            active_tasks_.fetch_sub(1, std::memory_order_relaxed);
            if (processed == BatchSize) ScheduleConsumer();
        }

    public:
        AsyncQueue(size_t capacity, size_t max_msg_size, size_t max_bin_size, CBE::UC::UC_ThreadPool::SmartThreadPool* pool, OverflowPolicy policy, std::function<void(LogRecord&)> processor)
            : buffer_(capacity), capacity_mask_(capacity - 1), max_msg_size_(max_msg_size), max_attrs_(16), max_bin_size_(max_bin_size),
            pool_(pool), policy_(policy), processor_(std::move(processor)) {

#ifndef LSTREAM_DISABLE_VALIDATION
            if ((capacity & (capacity - 1)) != 0) throw std::invalid_argument("AsyncQueue capacity must be a power of 2");
#endif

            message_pool_ = std::make_unique<char[]>(capacity * max_msg_size_);
            attr_pool_ = std::make_unique<Attribute[]>(capacity * max_attrs_);
            binary_pool_ = std::make_unique<std::byte[]>(capacity * max_bin_size_);

            for (size_t i = 0; i < capacity; ++i) {
                buffer_[i].sequence.store(i, std::memory_order_relaxed);
                buffer_[i].message_buf = message_pool_.get() + i * max_msg_size_;
                buffer_[i].attributes = attr_pool_.get() + i * max_attrs_;
                buffer_[i].binary_buf = binary_pool_.get() + i * max_bin_size_;
            }
        }

        LSTREAM_FORCE_INLINE void Push(const LogRecord& record) {
            size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
            for (;;) {
                Slot* slot = &buffer_[pos & capacity_mask_];
                size_t seq = slot->sequence.load(std::memory_order_acquire);
                intptr_t dif = (intptr_t)seq - (intptr_t)pos;

                if (dif == 0) [[likely]] {
                    if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                        slot->metadata = record.metadata;

                        slot->message_len = std::min(record.formatted_message.size(), max_msg_size_);
                        std::memcpy(slot->message_buf, record.formatted_message.data(), slot->message_len);

                        slot->attr_count = std::min(record.attributes.size() + record.injected_count, max_attrs_);
                        size_t idx = 0;
                        for (const auto& attr : record.attributes) {
                            if (idx < slot->attr_count) slot->attributes[idx++] = attr;
                        }
                        for (size_t i = 0; i < record.injected_count; ++i) {
                            if (idx < slot->attr_count) slot->attributes[idx++] = record.injected_attributes[i];
                        }

                        slot->binary_len = std::min(record.binary_payload.size(), max_bin_size_);
                        std::memcpy(slot->binary_buf, record.binary_payload.data(), slot->binary_len);

                        slot->sequence.store(pos + 1, std::memory_order_release);
                        ScheduleConsumer();
                        return;
                    }
                }
                else if (dif < 0) [[unlikely]] {
                    if (policy_ == OverflowPolicy::Drop && record.metadata.severity < Severity::Error) {
                        dropped_count_.fetch_add(1, std::memory_order_relaxed);
                        return;
                    }
                    processor_(const_cast<LogRecord&>(record));
                    return;
                }
                else pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
    };

    // ============================================================================
    // PIPELINE LAYER: STAGES
    // ============================================================================
    class IPipelineStage {
    public:
        virtual ~IPipelineStage() = default;
        virtual bool Process(LogRecord& record) const = 0;
    };

    class Pipeline {
        std::vector<std::shared_ptr<IPipelineStage>> stages_;
    public:
        void AddStage(std::shared_ptr<IPipelineStage> stage) { stages_.push_back(std::move(stage)); }
        bool Process(LogRecord& record) const {
            for (auto& stage : stages_) {
                if (!stage->Process(record)) [[unlikely]] return false;
            }
            return true;
        }
    };

    class FormattingStage : public IPipelineStage {
        std::shared_ptr<IFormatter> formatter_;
    public:
        explicit FormattingStage(std::shared_ptr<IFormatter> fmt) : formatter_(std::move(fmt)) {}
        bool Process(LogRecord& record) const override {
            formatter_->Format(record);
            return true;
        }
    };

    // ============================================================================
    // NAMESPACE TREE & REGISTRY (WITH LIVE RELOAD)
    // ============================================================================
    enum class ExecutionMode { Synchronous, Asynchronous };

    struct NodeConfig {
        Pipeline pipeline;
        std::vector<std::shared_ptr<ISink>> sinks;
        std::shared_ptr<AsyncQueue> async_queue;
        ExecutionMode mode = ExecutionMode::Synchronous;
    };

    template <typename Mutex = std::mutex, typename SharedMutex = std::shared_mutex>
    class LoggerNodeImpl {
        std::string name_;
        LoggerNodeImpl* parent_;
        std::atomic<Severity> min_severity_{ Severity::Info };
        std::atomic<bool> inherit_{ true };

        SharedMutex config_mutex_;
        std::shared_ptr<const NodeConfig> config_;

    public:
        LoggerNodeImpl(std::string name, LoggerNodeImpl* parent)
            : name_(std::move(name)), parent_(parent), config_(std::make_shared<NodeConfig>()) {
        }

        void SetSeverity(Severity sev) { min_severity_.store(sev, std::memory_order_relaxed); }
        void SetInheritance(bool inherit) { inherit_.store(inherit, std::memory_order_relaxed); }

        void SetConfig(std::shared_ptr<const NodeConfig> new_config) {
            std::unique_lock<SharedMutex> lock(config_mutex_);
            config_ = std::move(new_config);
        }

        LSTREAM_FORCE_INLINE std::shared_ptr<const NodeConfig> GetConfig() {
            std::shared_lock<SharedMutex> lock(config_mutex_);
            return config_;
        }

        LSTREAM_FORCE_INLINE Severity GetMinSeverity() const { return min_severity_.load(std::memory_order_relaxed); }
        const std::string& GetName() const { return name_; }

        LSTREAM_FORCE_INLINE void Process(LogRecord& record) {
            auto cfg = GetConfig();
            if (cfg->mode == ExecutionMode::Asynchronous && cfg->async_queue) [[likely]] {
                cfg->async_queue->Push(record);
            }
            else {
                ProcessSync(record, cfg);
            }
        }

        void ProcessSync(LogRecord& record, std::shared_ptr<const NodeConfig> cfg = nullptr) {
            if (record.metadata.severity < min_severity_.load(std::memory_order_relaxed)) [[unlikely]] return;
            if (!cfg) cfg = GetConfig();

            std::string_view original_msg = record.formatted_message;

            if (cfg->pipeline.Process(record)) [[likely]] {
                for (auto& sink : cfg->sinks) sink->Log(record);
            }

            record.formatted_message = original_msg;

            if (inherit_.load(std::memory_order_relaxed) && parent_) {
                parent_->ProcessSync(record);
            }
        }
    };

    template <typename Mutex = std::mutex, typename SharedMutex = std::shared_mutex>
    class LoggerRegistryImpl {
        SharedMutex mutex_;
        std::map<std::string, std::shared_ptr<LoggerNodeImpl<Mutex, SharedMutex>>> nodes_;

        std::shared_ptr<LoggerNodeImpl<Mutex, SharedMutex>> GetNodeUnlocked(const std::string& name) {
            auto it = nodes_.find(name);
            if (it != nodes_.end()) return it->second;

            std::shared_ptr<LoggerNodeImpl<Mutex, SharedMutex>> parent = nullptr;
            auto pos = name.find_last_of('.');
            if (pos != std::string::npos) {
                parent = GetNodeUnlocked(name.substr(0, pos));
            }

            auto node = std::make_shared<LoggerNodeImpl<Mutex, SharedMutex>>(name, parent.get());
            if (!parent) {
                auto cfg = std::make_shared<NodeConfig>();
                cfg->pipeline.AddStage(std::make_shared<FormattingStage>(std::make_shared<PatternFormatter>("{color}[{timestamp}] [{level}] [{index}] {message}{reset}")));
                node->SetConfig(cfg);
            }

            nodes_[name] = node;
            return node;
        }

    public:
        static LoggerRegistryImpl& Instance() {
            static LoggerRegistryImpl instance;
            return instance;
        }

        std::shared_ptr<LoggerNodeImpl<Mutex, SharedMutex>> GetNode(const std::string& name) {
            {
                std::shared_lock<SharedMutex> lock(mutex_);
                auto it = nodes_.find(name);
                if (it != nodes_.end()) return it->second;
            }
            std::unique_lock<SharedMutex> lock(mutex_);
            return GetNodeUnlocked(name);
        }
    };

    // ============================================================================
    // JSON CONFIGURATION MANAGER
    // ============================================================================
    template <typename Mutex = std::mutex, typename SharedMutex = std::shared_mutex>
    class JsonConfigParser {
        CBE::UC::UC_ThreadPool::SmartThreadPool* thread_pool_ = nullptr;
        std::map<std::string, std::shared_ptr<ISink>> parsed_sinks_;

        std::shared_ptr<ISink> CreateSink(const rapidjson::Value& val) {
            if (!val.IsObject() || !val.HasMember("type") || !val["type"].IsString()) {
                throw std::runtime_error("Invalid sink configuration");
            }
            std::string type = val["type"].GetString();
            if (type == "console") {
                return std::make_shared<ConsoleSinkImpl<Mutex>>();
            }
            else if (type == "rotating_file") {
                if (!val.HasMember("pattern") || !val["pattern"].IsString()) throw std::runtime_error("Rotating file sink requires a 'pattern' string");
                return std::make_shared<RotatingFileSinkImpl<Mutex>>(
                    val["pattern"].GetString(),
                    val.HasMember("max_size") ? val["max_size"].GetUint64() : 1048576,
                    val.HasMember("max_files") ? val["max_files"].GetUint() : 5
                );
            }
            else if (type == "binary") {
                if (!val.HasMember("filename") || !val["filename"].IsString()) throw std::runtime_error("Binary sink requires a 'filename' string");
                bool write_meta = val.HasMember("write_metadata") && val["write_metadata"].IsBool() ? val["write_metadata"].GetBool() : true;
                return std::make_shared<BinarySinkImpl<Mutex>>(val["filename"].GetString(), write_meta);
            }
            throw std::runtime_error("Unknown sink type: " + type);
        }

        std::shared_ptr<IFormatter> CreateFormatter(const rapidjson::Value& val) {
            if (!val.IsObject() || !val.HasMember("type") || !val["type"].IsString()) {
                throw std::runtime_error("Invalid formatter configuration");
            }
            std::string type = val["type"].GetString();
            if (type == "pattern") {
                if (!val.HasMember("pattern") || !val["pattern"].IsString()) throw std::runtime_error("Pattern formatter requires a 'pattern' string");
                auto fmt = std::make_shared<PatternFormatter>(val["pattern"].GetString());
                if (val.HasMember("timestamp_format") && val["timestamp_format"].IsString()) {
                    fmt->SetTimestampFormat(val["timestamp_format"].GetString());
                }
                return fmt;
            }
            else if (type == "json") {
                return std::make_shared<JsonFormatter>();
            }
            else if (type == "plain") {
                return std::make_shared<PlainTextFormatter>();
            }
            else if (type == "binary") {
                return std::make_shared<BinaryFormatter>();
            }
            throw std::runtime_error("Unknown formatter type: " + type);
        }

    public:
        void SetThreadPool(CBE::UC::UC_ThreadPool::SmartThreadPool* pool) { thread_pool_ = pool; }

        void ApplyConfig(const std::string& json_str) {
            rapidjson::Document doc;
            doc.Parse(json_str.c_str());
#ifndef LSTREAM_DISABLE_VALIDATION
            if (doc.HasParseError()) throw std::runtime_error("JSON Parse Error");
#endif

            if (doc.HasMember("sinks")) {
                for (auto& m : doc["sinks"].GetObj()) {
                    parsed_sinks_[m.name.GetString()] = CreateSink(m.value);
                }
            }

            if (doc.HasMember("loggers")) {
                for (auto& m : doc["loggers"].GetObj()) {
                    std::string logger_name = m.name.GetString();
                    auto node = LoggerRegistryImpl<Mutex, SharedMutex>::Instance().GetNode(logger_name);
                    const auto& val = m.value;

                    if (val.HasMember("level")) node->SetSeverity(ParseSeverity(val["level"].GetString()));
                    if (val.HasMember("inherit")) node->SetInheritance(val["inherit"].GetBool());

                    auto new_cfg = std::make_shared<NodeConfig>();

                    if (val.HasMember("format")) {
                        new_cfg->pipeline.AddStage(std::make_shared<FormattingStage>(CreateFormatter(val["format"])));
                    }

                    if (val.HasMember("sinks")) {
                        for (auto& s : val["sinks"].GetArray()) {
                            std::string sname = s.GetString();
                            if (parsed_sinks_.count(sname)) {
                                new_cfg->sinks.push_back(parsed_sinks_[sname]);
                            }
                        }
                    }

                    if (val.HasMember("async") && val["async"].HasMember("enabled") && val["async"]["enabled"].GetBool()) {
                        new_cfg->mode = ExecutionMode::Asynchronous;
                        size_t q_size = val["async"].HasMember("queue_size") ? val["async"]["queue_size"].GetUint64() : 8192;
                        size_t bin_size = val["async"].HasMember("binary_size") ? val["async"]["binary_size"].GetUint64() : 1024;
                        OverflowPolicy pol = OverflowPolicy::Block;
                        if (val["async"].HasMember("overflow") && std::string(val["async"]["overflow"].GetString()) == "drop") {
                            pol = OverflowPolicy::Drop;
                        }
                        new_cfg->async_queue = std::make_shared<AsyncQueue>(q_size, 1024, bin_size, thread_pool_, pol, [node](LogRecord& rec) {
                            node->ProcessSync(rec);
                            });
                    }

                    node->SetConfig(new_cfg);
                }
            }
        }
    };

    // ============================================================================
    // FRONT-END: LOGGER CLASS
    // ============================================================================
    template <typename Mutex = std::mutex, typename SharedMutex = std::shared_mutex>
    class LoggerImpl {
        std::shared_ptr<LoggerNodeImpl<Mutex, SharedMutex>> node_;

    public:
        explicit LoggerImpl(const std::string& ns)
            : node_(LoggerRegistryImpl<Mutex, SharedMutex>::Instance().GetNode(ns)) {
        }

        template <typename... Args>
        LSTREAM_FORCE_INLINE void Log(Severity sev, const MultiIndex& idx, std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) {
            if (sev < node_->GetMinSeverity()) [[unlikely]] return;

            std::array<char, 1024> stack_buf;
            auto result = std::format_to_n(stack_buf.data(), stack_buf.size() - 1, fmt.fmt, std::forward<Args>(args)...);
            *result.out = '\0';

            std::string owned_dynamic_msg;
            std::string_view msg;

            if (result.size > static_cast<decltype(result.size)>(stack_buf.size() - 1)) {
                owned_dynamic_msg.resize(result.size);
                std::format_to_n(owned_dynamic_msg.data(), result.size, fmt.fmt, std::forward<Args>(args)...);
                msg = owned_dynamic_msg;
            }
            else {
                msg = std::string_view(stack_buf.data(), result.size);
            }

            LogMetadata meta{ sev, fmt.loc, std::chrono::system_clock::now(), std::this_thread::get_id(), node_->GetName(), idx };
            LogRecord record;
            record.metadata = meta;
            record.formatted_message = msg;

            if (!owned_dynamic_msg.empty()) {
                record.owned_message = std::move(owned_dynamic_msg);
                record.formatted_message = record.owned_message;
            }

            node_->Process(record);
        }

        template <typename... Args>
        LSTREAM_FORCE_INLINE void Log(Severity sev, std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) {
            Log(sev, MultiIndex{}, fmt, std::forward<Args>(args)...);
        }

        template <typename T>
        LSTREAM_FORCE_INLINE void LogObject(Severity sev, const MultiIndex& idx, const T& obj, std::string_view description = "", const SourceLoc& loc = SourceLoc::current()) {
            if (sev < node_->GetMinSeverity()) [[unlikely]] return;

            size_t required_size = Serializer<T>::Measure(obj);

            LogMetadata meta{ sev, loc, std::chrono::system_clock::now(), std::this_thread::get_id(), node_->GetName(), idx };
            LogRecord record;
            record.metadata = meta;
            record.formatted_message = description;

            if (required_size <= record.stack_binary_buf.size()) [[likely]] {
                Serializer<T>::Serialize(obj, record.stack_binary_buf.data());
                record.binary_payload = std::span<const std::byte>(record.stack_binary_buf.data(), required_size);
            }
            else {
                record.owned_payload.resize(required_size);
                Serializer<T>::Serialize(obj, record.owned_payload.data());
                record.binary_payload = record.owned_payload;
            }

            node_->Process(record);
        }

        template <typename T>
        LSTREAM_FORCE_INLINE void LogObject(Severity sev, const T& obj, std::string_view description = "", const SourceLoc& loc = SourceLoc::current()) {
            LogObject(sev, MultiIndex{}, obj, description, loc);
        }

        template <typename... Args> LSTREAM_FORCE_INLINE void Trace(const MultiIndex& idx, std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Trace, idx, fmt, std::forward<Args>(args)...); }
        template <typename... Args> LSTREAM_FORCE_INLINE void Debug(const MultiIndex& idx, std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Debug, idx, fmt, std::forward<Args>(args)...); }
        template <typename... Args> LSTREAM_FORCE_INLINE void Info(const MultiIndex& idx, std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Info, idx, fmt, std::forward<Args>(args)...); }
        template <typename... Args> LSTREAM_FORCE_INLINE void Warn(const MultiIndex& idx, std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Warning, idx, fmt, std::forward<Args>(args)...); }
        template <typename... Args> LSTREAM_FORCE_INLINE void Error(const MultiIndex& idx, std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Error, idx, fmt, std::forward<Args>(args)...); }
        template <typename... Args> LSTREAM_FORCE_INLINE void Fatal(const MultiIndex& idx, std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Critical, idx, fmt, std::forward<Args>(args)...); }

        template <typename... Args> LSTREAM_FORCE_INLINE void Trace(std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Trace, MultiIndex{}, fmt, std::forward<Args>(args)...); }
        template <typename... Args> LSTREAM_FORCE_INLINE void Debug(std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Debug, MultiIndex{}, fmt, std::forward<Args>(args)...); }
        template <typename... Args> LSTREAM_FORCE_INLINE void Info(std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Info, MultiIndex{}, fmt, std::forward<Args>(args)...); }
        template <typename... Args> LSTREAM_FORCE_INLINE void Warn(std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Warning, MultiIndex{}, fmt, std::forward<Args>(args)...); }
        template <typename... Args> LSTREAM_FORCE_INLINE void Error(std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Error, MultiIndex{}, fmt, std::forward<Args>(args)...); }
        template <typename... Args> LSTREAM_FORCE_INLINE void Fatal(std::type_identity_t<LogFormatString<Args...>> fmt, Args&&... args) { Log(Severity::Critical, MultiIndex{}, fmt, std::forward<Args>(args)...); }
    };

    using Logger = LoggerImpl<std::mutex, std::shared_mutex>;
    using ConfigManager = JsonConfigParser<std::mutex, std::shared_mutex>;
    using BinarySink = BinarySinkImpl<std::mutex>;

    // Zero-Overhead Aliases (For Benchmarking)
    using UnlockedLogger = LoggerImpl<NullMutex, NullSharedMutex>;
    using UnlockedConsoleSink = ConsoleSinkImpl<NullMutex>;
    using UnlockedRotatingFileSink = RotatingFileSinkImpl<NullMutex>;
    using UnlockedBinarySink = BinarySinkImpl<NullMutex>;

} // namespace Lstream

// ============================================================================
// CUSTOM FORMATTER SPECIALIZATIONS
// ============================================================================
template <typename F>
struct std::formatter<Lstream::LazyEval<F>> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }
    auto format(const Lstream::LazyEval<F>& lazy, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", lazy.func());
    }
};