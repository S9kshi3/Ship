#pragma once
#include <asio.hpp>
#include <vector>
#include <queue>
#include <thread>
#include <functional>
#include <future>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <utility>

namespace CBE {
    namespace UC {
        namespace UC_ThreadPool {
            namespace UC_Network {

                /**
                 * @brief A self-contained thread pool that owns and runs its own io_context.
                 *
                 * This class is designed to run Asio's asynchronous event loop on a fixed
                 * number of threads, making it ideal for network I/O. It provides a simple
                 * API for enqueuing tasks via asio::post().
                 */
                class SmartThreadPool {
                public:
                    /**
                     * @brief Constructs the thread pool and prepares it for activation.
                     * @param num_threads The number of worker threads to create.
                     */
                    explicit SmartThreadPool(size_t num_threads)
                        : io_context_(std::make_shared<asio::io_context>()),
                        work_guard_(asio::make_work_guard(*io_context_)),
                        num_threads_(num_threads),
                        is_running_(false) {
                        // We create the io_context and the work guard here, but
                        // the threads are started in activate().
                    };
                    // ----------------------------------------------------------------------
                    // Constructor 2: Adapter (uses an external io_context)
                    // ----------------------------------------------------------------------
                    /**
                        * @brief Constructs a thread pool that uses an external io_context.
                        *
                        * @param io_context A reference to the external io_context to use.
                        * @param num_threads The number of worker threads to create.
                     */
                    explicit SmartThreadPool(asio::io_context& io_context, size_t num_threads)
                        : io_context_(std::shared_ptr<asio::io_context>(&io_context, [](asio::io_context*) {})),
                        work_guard_(asio::make_work_guard(*io_context_)),
                        num_threads_(num_threads),
                        is_running_(false) {
                    }

                    explicit SmartThreadPool(std::shared_ptr<asio::io_context> shared_io_context, size_t num_threads)
                        : io_context_(shared_io_context),
                        work_guard_(asio::make_work_guard(*io_context_)),
                        num_threads_(num_threads),
                        is_running_(false) {
                    }
                    // The class is not copyable or movable.
                    SmartThreadPool(const SmartThreadPool&) = delete;
                    SmartThreadPool& operator=(const SmartThreadPool&) = delete;

                    /**
                     * @brief Destructor that gracefully shuts down the thread pool.
                     */
                    ~SmartThreadPool() {
                        stop();
                    };

                    /**
                     * @brief Starts the thread pool by running the io_context.
                     *
                     * This method creates the worker threads and starts the io_context's
                     * event loop on each of them. It should be called once after
                     * the thread pool object is created.
                     */
                    void activate() {
                        if (!is_running_) {
                            for (size_t i = 0; i < num_threads_; ++i) {
                                threads_.emplace_back([this]() {
                                    io_context_->run();
                                    });
                            }
                            is_running_ = true;
                        }
                    };

                    /**
                     * @brief Stops the thread pool and waits for all threads to join.
                     *
                     * This method can be called manually to initiate a graceful shutdown.
                     * It's also called automatically by the destructor.
                     */
                    void stop() {
                        if (is_running_) {
                            // First, reset the work guard to allow io_context.run() to exit
                            work_guard_.reset();

                            // Then, wait for all threads to finish their current tasks and exit
                            for (auto& thread : threads_) {
                                if (thread.joinable()) {
                                    thread.join();
                                }
                            }
                            is_running_ = false;
                        }
                    };

                    /**
                     * @brief Enqueues a task to be executed asynchronously by a worker thread.
                     * @tparam F The type of the callable task.
                     * @param task The callable task to be executed.
                     */
                    template <typename F>
                    void enqueue(F&& task) {
                        if (is_running_) {
                            asio::post(*io_context_, std::forward<F>(task));
                        }
                    }

                    /**
                     * @brief Returns a reference to the io_context.
                     * This is needed to register asynchronous operations with this thread pool.
                     */
                    asio::io_context& get_io_context() {
                        if (!io_context_) {
                            // This should not happen in a correctly running program
                            throw std::runtime_error("io_context is not initialized");
                        }
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

            /**
             * @brief A traditional thread pool using std::mutex and std::condition_variable.
             *
             * This class is suitable for general-purpose, CPU-bound tasks where you want
             * to manage a pool of workers and a task queue manually.
             */
            class SmartThreadPool {
            public:
                SmartThreadPool(size_t num_threads = std::thread::hardware_concurrency())
                    : stop_(false) {
                    if (num_threads == 0) {
                        num_threads = 1;
                    }
                    for (size_t i = 0; i < num_threads; ++i) {
                        workers_.emplace_back(
                            [this] {
                                while (true) {
                                    std::function<void()> task;

                                    {
                                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                                        this->condition_.wait(lock, [this] {
                                            return this->stop_ || !this->tasks_.empty();
                                            });

                                        // Condition to exit the worker loop: stop is true and no more tasks
                                        if (this->stop_ && this->tasks_.empty()) {
                                            return;
                                        }
                                        task = std::move(this->tasks_.front());
                                        this->tasks_.pop();

                                    } // 'lock' goes out of scope here, and the mutex is released.

                                    try {
                                        task();
                                    }
                                    catch (...) {
                                        // Log or handle exceptions from tasks if necessary
                                    }
                                }
                            });
                    }
                    
                };

                // No need for a copy/move constructor since std::vector<std::thread> is not copyable
                SmartThreadPool(const SmartThreadPool&) = delete;
                SmartThreadPool& operator=(const SmartThreadPool&) = delete;

                ~SmartThreadPool() {
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        stop_ = true;
                    }
                    condition_.notify_all();
                    for (std::thread& worker : workers_) {
                        if (worker.joinable()) {
                            worker.join();
                        }
                    }
                }

                template<class F, class... Args>
                auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
                    using return_type = std::invoke_result_t<F, Args...>;

                    auto task = std::make_shared<std::packaged_task<return_type()>>(
                        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
                    );

                    std::future<return_type> res = task->get_future();
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        if (stop_) {
                            throw std::runtime_error("enqueue on stopped ThreadPool");
                        }
                        tasks_.emplace([task]() { (*task)(); });
                    }
                    condition_.notify_one();
                    return res;
                }

            private:
                std::vector<std::thread> workers_;
                std::queue<std::function<void()>> tasks_;
                std::mutex queue_mutex_;
                std::condition_variable condition_;
                bool stop_;
            };
        } // namespace UC_ThreadPool
    } // namespace UC
} // namespace CBE