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
#include <cstdio>


/**
 * @brief Provides a flexible and efficient logging framework with multiple storage modes.
 * @details The Lstream namespace encapsulates a versatile `Log` class designed for applications
 * requiring various logging behaviors, including in-memory caching, direct file writing,
 * and a hybrid file-backed caching mechanism. It supports custom log entry types via templates,
 * tracks timestamps, and calculates time intervals between log events.
 */
namespace Lstream {

    /**
     * @brief Defines the operational modes for the Log class.
     */
    enum class LogMode {
        /** @brief Logs are stored exclusively in RAM, providing high performance. */
        File,
        /** @brief Logs are primarily written to a file on disk. In-memory storage is minimal, used as a temporary buffer. */
        Cache,
        /** @brief Logs are written to a file and a limited number are also kept in an in-memory cache for fast retrieval. */
        FileCache
    };

    /**
     * @brief A templated class for managing application logs with various storage strategies.
     * @details The `Log` class offers flexible logging capabilities. It can operate in three distinct modes:
     * - `LogMode::Cache`: All log entries are stored in a dynamic in-memory vector.
     * - `LogMode::File`: Log entries are primarily persisted to a specified file, with minimal in-memory buffering.
     * - `LogMode::FileCache`: Combines file persistence with an LRU (Least Recently Used) in-memory cache
     * for frequently accessed log entries, optimizing retrieval performance.
     * The class supports automatic resizing of its in-memory capacity and tracks timestamps
     * and time intervals between log operations.
     *
     * @tparam T The type of data to be stored as a log entry. This can be any type that supports
     * default construction, assignment, and stream insertion (`operator<<`). For `std::vector<std::string>`,
     * specializations handle multi-line string messages.
     */
    template <class T>
    class Log {
    public:
        Log();
    private:
        /** @brief Stores the index of the last log entry that was accessed (read or written). */
        mutable unsigned int last_accessed_log_index_;
        /** @brief Stores the highest logical index currently occupied by a log entry. */
        mutable unsigned int current_last_occupied_log_index_;
        /** @brief A high-resolution time point used as a reference for calculating time intervals between log events. */
        std::chrono::high_resolution_clock::time_point last_log_timestamp_;

        /** @brief The maximum number of log entries that can be stored in memory (`Svalue_`). */
        int size_;
        /** @brief The current operational mode of the logging system (File, Cache, or FileCache). */
        LogMode mode_;
        /** @brief The name of the file used for logging when in `File` or `FileCache` mode. */
        std::string filename_;
        /** @brief The file stream object used for reading from and writing to the log file. */
        mutable std::fstream log_file_;

        /** @brief A deque (double-ended queue) used as the in-memory LRU cache in `FileCache` mode.
         * Stores pairs of (log_index, log_value). */
        std::deque<std::pair<unsigned int, T>> file_cache_queue_;
        /** @brief The maximum capacity of the `file_cache_queue_` in `FileCache` mode. */
        int file_cache_size_;

        /** @brief A map storing the byte offsets of each main log entry within the log file.
         * Used for fast seeking when retrieving log entries from disk. */
        std::map<unsigned int, long long> file_index_offsets_;

        /**
         * @brief Updates the internal `last_accessed_log_index_` to the provided index.
         * @param index The index of the log entry that was just accessed.
         */
        void UpdateLastLogIndex(unsigned int index) const;

        /**
         * @brief Retrieves the current system time formatted as "HH:MM:SS AM/PM".
         * @return A string representation of the current system time.
         */
        std::string GetTimeL() const;

        /**
         * @brief Formats a given `std::chrono::system_clock::time_point` as "HH:MM:SS AM/PM".
         * @param tp The time point to format.
         * @return A string representation of the specified time point.
         */
        std::string GetTimeL(std::chrono::system_clock::time_point tp) const;

        /**
         * @brief Calculates the time interval between the current time and the `last_log_timestamp_`.
         * @details Updates `last_log_timestamp_` to the current time for subsequent interval calculations.
         * The interval is formatted as milliseconds (MS) if less than 1 second, otherwise as seconds (S).
         * @return A string representing the calculated time interval.
         */
        std::string GetFormattedTimeInterval();

        /**
         * @brief Formats a single part of a log line into a standardized string.
         * @tparam U The type of the message part (e.g., `std::string` for `std::vector<std::string>` or `T` for generic types).
         * @param main_index The main logical index of the log entry.
         * @param sub_index The sub-index for multi-part log entries (e.g., lines in a `std::vector<std::string>`).
         * @param message_part The actual message content for this part of the log entry.
         * @param time_str The formatted timestamp string.
         * @param interval_str The formatted time interval string.
         * @param status_char A character indicating the log status (e.g., '+' for success, '-' for error).
         * @return The fully formatted log line string.
         */
        template<typename U>
        std::string FormatLogLine(unsigned int main_index, unsigned int sub_index, const U& message_part, const std::string& time_str, const std::string& interval_str, char status_char);

        /**
         * @brief Writes a single formatted log entry line to the specified log file.
         * @details This internal function handles file opening/reopening and writes the formatted string.
         * For `std::vector<std::string>`, this function is called for each string in the vector.
         * It also records the file offset for the main index.
         * @param main_index The main logical index of the log entry.
         * @param sub_index The sub-index for multi-part log entries.
         * @param message_to_write The string content to write to the file.
         * @param time_str The formatted timestamp string.
         * @param interval_str The formatted time interval string.
         * @param status_char A character indicating the log status.
         * @throw std::runtime_error if the log file cannot be opened for writing.
         */
        void WriteLogEntryToFile(unsigned int main_index, unsigned int sub_index, const std::string& message_to_write, const std::string& time_str, const std::string& interval_str, char status_char);

        /**
         * @brief Reads a complete log entry from the file based on its main index.
         * @details This function is responsible for retrieving the raw log data from the physical file.
         * It handles reopening the file in input mode and seeking to the correct position.
         * For `std::string` types, it reads the entire line to prevent truncation.
         * @param main_index The logical index of the log entry to read.
         * @return The log entry data of type `T`.
         * @throw std::logic_error if called in `Cache` mode (as logs are not file-based).
         * @throw std::runtime_error if the log file cannot be found, opened for reading, or if the entry is not found.
         */
        T ReadLogEntryFromFile(unsigned int main_index) const;

        /**
         * @brief Adds a log entry to the in-memory LRU cache (`file_cache_queue_`) for `FileCache` mode.
         * @details If the entry already exists, it's moved to the front (MRU). If the cache exceeds its size limit,
         * the least recently used entry is evicted from the back.
         * @param index The logical index of the log entry.
         * @param value The log entry value to add.
         */
        void AddToCache(unsigned int index, const T& value);

        /**
         * @brief Attempts to retrieve a log entry from the in-memory LRU cache in a const context.
         * @details This function only checks if an entry is present in the cache without affecting its MRU status.
         * @param index The logical index of the log entry to retrieve.
         * @param out_value A reference where the retrieved value will be placed if found.
         * @return True if the entry was found in the cache, false otherwise.
         */
        bool GetFromCache(unsigned int index, T& out_value) const;

        /**
         * @brief Attempts to retrieve a log entry from the in-memory LRU cache, updating its MRU status.
         * @details If the entry is found, it is moved to the front of the cache (most recently used).
         * This is typically called by non-const `RetriveLog` to manage the cache order.
         * @param index The logical index of the log entry to retrieve.
         * @param out_value A reference where the retrieved value will be placed if found.
         * @return True if the entry was found in the cache, false otherwise.
         */
        bool GetFromCacheMutable(unsigned int index, T& out_value);

    public:
        /** @brief The primary in-memory storage for log entries. Its usage varies based on `LogMode`.
         * In `Cache` and `FileCache` modes, it holds the actual log data.
         * In `File` mode, it acts as a minimal temporary buffer for `RetriveLog` returns. */
        std::vector<T> Svalue_;
        /** @brief A vector of characters used to track the occupancy status of each log index in `Svalue_`.
         * `1` indicates occupied, `0` indicates free. (Used instead of `bool` for predictable behavior). */
        std::vector<char> Occupied_status_;

        /**
         * @brief Constructs a new `Log` object with specified capacity and operating mode.
         * @param sz The initial capacity for in-memory storage (`Svalue_`). For `File` mode, this is a minimal buffer size.
         * @param mode The logging mode (`LogMode::Cache`, `LogMode::File`, or `LogMode::FileCache`). Defaults to `LogMode::Cache`.
         * @param fname The filename to use for logging when in `File` or `FileCache` mode. Defaults to "application_log.txt".
         * @param cache_sz The size of the in-memory LRU cache for `FileCache` mode. Defaults to 10.
         * @throw std::invalid_argument if `sz` or `cache_sz` is negative.
         */
        Log(int sz, LogMode mode = LogMode::Cache, const std::string& fname = "application_log.txt", int cache_sz = 10);

        /**
         * @brief Destructor for the `Log` object.
         * @details Ensures that the log file stream is properly closed to prevent data loss or resource leaks.
         */
        ~Log();

        /**
         * @brief Checks if a given log index is within the valid bounds of the in-memory storage (`Svalue_`).
         * @param index The index to check.
         * @return True if the index is within bounds, false otherwise.
         */
        bool DoBoundCheck(unsigned int index) const;

        /**
         * @brief Adds a new log entry at the next available logical position.
         * @details This overload automatically determines the next unoccupied index. If `auto_resize` is true
         * and the current capacity is exhausted, the in-memory storage will be resized.
         * @param value The data for the log entry.
         * @param status_char A character to denote the status (e.g., '+' for success, '-' for error, '!' for warning).
         * @param auto_resize If true, the internal `Svalue_` vector will automatically grow if full. Defaults to true.
         * @return True if the log entry was successfully set.
         * @throw std::out_of_range if the log is full and `auto_resize` is false.
         */
        bool SetLog(const T& value, char status_char, bool auto_resize = true);

        /**
         * @brief Adds a new log entry at a specific logical index.
         * @details This function places the log `value` at the specified `index`.
         * It updates internal tracking for occupied status and the last occupied index.
         * @param value The data for the log entry.
         * @param index The specific logical index where the log entry should be placed.
         * @param status_char A character to denote the status.
         * @return True if the log entry was successfully set.
         * @throw std::out_of_range if the `index` is beyond the current bounds of `Svalue_`
         * (when in `Cache` or `FileCache` mode) and auto-resize is not implicitly handled.
         */
        bool SetLog(const T& value, unsigned int index, char status_char);

        /**
         * @brief Checks if a log entry at a given index is currently occupied.
         * @details For `Cache` and `FileCache` modes, it checks the `Occupied_status_` vector.
         * For `File` mode, it checks if an offset for the index is recorded in `file_index_offsets_`.
         * @param index The logical index to check.
         * @return True if the log entry is occupied, false otherwise.
         */
        bool IsOccupied(unsigned int index) const;

        /**
         * @brief Retrieves a log entry by its logical index.
         * @details This function retrieves the log data. Behavior varies by mode:
         * - `Cache`: Retrieves directly from `Svalue_`.
         * - `FileCache`: Attempts retrieval from `file_cache_queue_` first (updating LRU status),
         * then falls back to `ReadLogEntryFromFile` if not cached.
         * - `File`: Retrieves directly from `ReadLogEntryFromFile`.
         * The retrieved value is temporarily stored in `Svalue_` if not already there, and a reference is returned.
         * @param index The logical index of the log entry to retrieve.
         * @return A non-const reference to the retrieved log entry.
         * @throw std::out_of_range if the index is out of bounds for the in-memory storage.
         * @throw std::runtime_error if no log entry is found at the given index (either not occupied or not found in file).
         */
        T& RetriveLog(unsigned int index);

        /**
         * @brief Retrieves a log entry by its logical index in a constant context.
         * @details Similar to the non-const `RetriveLog`, but returns a const reference.
         * In `FileCache` mode, it still updates the LRU status internally via `const_cast`
         * for cache management purposes, which is typically acceptable for internal optimizations.
         * @param index The logical index of the log entry to retrieve.
         * @return A const reference to the retrieved log entry.
         * @throw std::out_of_range if the index is out of bounds for the in-memory storage.
         * @throw std::runtime_error if no log entry is found at the given index.
         */
        const T& RetriveLog(unsigned int index) const;

        /**
         * @brief Clears a log entry, marking its index as unoccupied.
         * @details In `Cache` and `FileCache` modes, the `Occupied_status_` for the index is set to `0` (false).
         * In `FileCache` mode, the entry is also removed from the `file_cache_queue_`.
         * Note: This operation does not physically remove data from a persistent file in `File` or `FileCache` mode.
         * @param index The logical index of the log entry to clear.
         */
        void ClearLog(unsigned int index);

        /**
         * @brief Resizes the in-memory storage (`Svalue_` and `Occupied_status_`) to a new capacity.
         * @details Existing log entries are moved to the new storage. Indices are adjusted if they fall
         * outside the new bounds. This operation is relevant for all modes, even if `Svalue_`
         * is a temporary buffer for `File` mode.
         * @param new_sz The new desired size for the in-memory storage.
         * @throw std::invalid_argument if `new_sz` is negative.
         */
        void Resize(int new_sz);

        /**
         * @brief Gets the index of the log entry that was most recently accessed (read or written).
         * @return The last accessed log index.
         */
        unsigned int GetLastAccessedLogIndex() const;

        /**
         * @brief Gets the highest logical index currently holding an occupied log entry.
         * @return The current last occupied log index.
         */
        unsigned int GetCurrentLastOccupiedLogIndex() const;

        /**
         * @brief Gets the current allocated size of the in-memory log storage (`Svalue_`).
         * @return The current size of the log.
         */
        int GetSize() const;

        /**
         * @brief Retrieves the current operational mode of the logger.
         * @return The `LogMode` currently in effect.
         */
        LogMode GetLogMode() const { return mode_; }

        /**
         * @brief Flushes any buffered log data to the underlying file.
         * @details This is relevant for `File` and `FileCache` modes to ensure data persistence.
         */
        void FlushFileLog();

        /**
         * @brief Explicitly sets the reference timestamp used for calculating subsequent time intervals.
         * @details This can be used to synchronize or reset the interval timer at specific points in the application lifecycle.
         * @param new_timestamp The new high-resolution time point to set as the reference.
         */
        void SetIntervalReferenceTimestamp(std::chrono::high_resolution_clock::time_point new_timestamp);

        /**
         * @brief Retrieves the `high_resolution_clock::time_point` that is currently used as the interval reference.
         * @return The `time_point` of the last interval calculation reference.
         */
        std::chrono::high_resolution_clock::time_point GetLastLogIntervalTimestamp() const;

        /**
         * @brief Retrieves the last interval reference timestamp formatted as "HH:MM:SS AM/PM".
         * @return A string representation of the last interval reference timestamp.
         */
        std::string GetFormattedLastLogIntervalTimestamp() const;
    };

    // --- Template Specializations for std::vector<std::string> ---
    /**
     * @brief Specialization of FormatLogLine for `std::vector<std::string>`.
     * @details This specialization formats a single string part from a multi-line log entry.
     * @tparam U Must be `std::string` for this specialization.
     * @param main_index The main logical index of the log entry.
     * @param sub_index The sub-index of the string within the `std::vector<std::string>`.
     * @param message_part The individual string message from the vector.
     * @param time_str The formatted timestamp.
     * @param interval_str The formatted time interval.
     * @param status_char The status character.
     * @return The formatted log line.
     */
    template <>
    template <>
    std::string Log<std::vector<std::string>>::FormatLogLine<std::string>(unsigned int main_index, unsigned int sub_index, const std::string& message_part, const std::string& time_str, const std::string& interval_str, char status_char) {
        std::stringstream ss;
        ss << main_index << ":" << sub_index << " [" << time_str << "] - [" << interval_str << "] ->[" << status_char << "] : " << message_part;
        return ss.str();
    }

    /**
     * @brief Specialization of WriteLogEntryToFile for `std::vector<std::string>`.
     * @details This function iterates over each string in the `std::vector<std::string>` and writes
     * each as a separate line in the log file, maintaining sub-indices.
     * It also records the starting file offset for the main log entry.
     * @param main_index The main logical index of the log entry.
     * @param sub_index The sub-index of the current string being written.
     * @param message_to_write The string message part to write.
     * @param time_str The formatted timestamp.
     * @param interval_str The formatted time interval.
     * @param status_char The status character.
     * @throw std::runtime_error if the log file cannot be opened for writing.
     */
    template <>
    void Log<std::vector<std::string>>::WriteLogEntryToFile(unsigned int main_index, unsigned int sub_index, const std::string& message_to_write, const std::string& time_str, const std::string& interval_str, char status_char) {
        if (mode_ == LogMode::File || mode_ == LogMode::FileCache) {
            if (!log_file_.is_open()) {
                log_file_.open(filename_, std::ios::out | std::ios::app);
                if (!log_file_.is_open()) {
                    throw std::runtime_error("Failed to open log file for writing: " + filename_);
                }
            }
            if (sub_index == 0) {
                file_index_offsets_[main_index] = log_file_.tellp();
            }
            log_file_ << FormatLogLine(main_index, sub_index, message_to_write, time_str, interval_str, status_char) << std::endl;
        }
    }

    /**
     * @brief Specialization of ReadLogEntryFromFile for `std::vector<std::string>`.
     * @details Reads all lines associated with a given main log index from the file and reconstructs
     * them into a `std::vector<std::string>`.
     * @param main_index The logical index of the log entry to read.
     * @return A `std::vector<std::string>` containing all message parts for the specified log entry.
     * @throw std::logic_error if called in `Cache` mode.
     * @throw std::runtime_error if the log file is not found, cannot be opened, or if the entry is not found.
     */
    template <>
    std::vector<std::string> Log<std::vector<std::string>>::ReadLogEntryFromFile(unsigned int main_index) const {
        std::vector<std::string> messages;
        if (mode_ == LogMode::Cache) {
            throw std::logic_error("ReadLogEntryFromFile called in Cache mode.");
        }

        if (log_file_.is_open()) {
            log_file_.close();
        }
        log_file_.open(filename_, std::ios::in);
        if (!log_file_.is_open()) {
            return messages;
        }

        log_file_.clear();
        if (file_index_offsets_.count(main_index)) {
            log_file_.seekg(file_index_offsets_.at(main_index));
        }
        else {
            log_file_.seekg(0, std::ios::beg);
        }


        std::string line;
        std::string target_prefix = std::to_string(main_index) + ":";

        while (std::getline(log_file_, line)) {
            if (line.rfind(target_prefix, 0) == 0) {
                size_t colon_pos = line.find(" : ");
                if (colon_pos != std::string::npos) {
                    messages.push_back(line.substr(colon_pos + 3));
                }
            }
            else if (!messages.empty()) {
                break;
            }
        }

        if (log_file_.is_open()) {
            log_file_.close();
        }
        log_file_.open(filename_, std::ios::out | std::ios::app);
        if (!log_file_.is_open()) {
            throw std::runtime_error("Failed to reopen log file for appending after read: " + filename_);
        }

        if (messages.empty() && !file_index_offsets_.count(main_index)) {
        }
        return messages;
    }

    // --- Generic Template Implementations for other T types ---
    /**
     * @brief Generic implementation to format a single part of a log line for non-`std::vector<std::string>` types.
     * @tparam T The type of the log entry.
     * @tparam U The type of the message part, typically `T`.
     * @param main_index The main logical index of the log entry.
     * @param sub_index The sub-index (typically 0 for single-line entries).
     * @param message_part The message content.
     * @param time_str The formatted timestamp.
     * @param interval_str The formatted time interval.
     * @param status_char The status character.
     * @return The fully formatted log line string.
     */
    template <class T>
    template <typename U>
    std::string Log<T>::FormatLogLine(unsigned int main_index, unsigned int sub_index, const U& message_part, const std::string& time_str, const std::string& interval_str, char status_char) {
        std::stringstream ss;
        ss << main_index << ":" << sub_index << " [" << time_str << "] - [" << interval_str << "] ->[" << status_char << "] : ";
        ss << message_part;
        return ss.str();
    }

    /**
     * @brief Generic implementation to write a log entry to the file for non-`std::vector<std::string>` types.
     * @details Records the file offset for the main log entry.
     * @tparam T The type of the log entry.
     * @param main_index The logical index of the log entry.
     * @param sub_index The sub-index (typically 0).
     * @param message_to_write The string representation of the log entry to write.
     * @param time_str The formatted timestamp.
     * @param interval_str The formatted time interval.
     * @param status_char The status character.
     * @throw std::runtime_error if the log file cannot be opened for writing.
     */
    template <class T>
    void Log<T>::WriteLogEntryToFile(unsigned int main_index, unsigned int sub_index, const std::string& message_to_write, const std::string& time_str, const std::string& interval_str, char status_char) {
        if (mode_ == LogMode::File || mode_ == LogMode::FileCache) {
            if (!log_file_.is_open()) {
                log_file_.open(filename_, std::ios::out | std::ios::app);
                if (!log_file_.is_open()) {
                    throw std::runtime_error("Failed to open log file for writing: " + filename_);
                }
            }
            if (sub_index == 0) {
                file_index_offsets_[main_index] = log_file_.tellp();
            }
            log_file_ << FormatLogLine(main_index, sub_index, message_to_write, time_str, interval_str, status_char) << std::endl;
        }
    }

    /**
     * @brief Generic implementation to read a log entry from the file for non-`std::vector<std::string>` types.
     * @details Reads the log entry based on its main index. Includes specialized logic for `std::string`
     * to read the entire line, preventing truncation.
     * @tparam T The type of the log entry.
     * @param main_index The logical index of the log entry to read.
     * @return The retrieved log entry of type `T`.
     * @throw std::logic_error if called in `Cache` mode.
     * @throw std::runtime_error if the log file is not found, cannot be opened, or if the entry is not found.
     */
    template <class T>
    T Log<T>::ReadLogEntryFromFile(unsigned int main_index) const {
        if (mode_ == LogMode::Cache) {
            throw std::logic_error("ReadLogEntryFromFile called in Cache mode.");
        }

        if (log_file_.is_open()) {
            log_file_.close();
        }
        log_file_.open(filename_, std::ios::in);
        if (!log_file_.is_open()) {
            throw std::runtime_error("Log file not found or could not be opened for reading: " + filename_);
        }

        log_file_.clear();
        if (file_index_offsets_.count(main_index)) {
            log_file_.seekg(file_index_offsets_.at(main_index));
        }
        else {
            log_file_.seekg(0, std::ios::beg);
        }

        std::string line;
        std::string target_prefix = std::to_string(main_index) + ":";
        T value_from_file = T();

        bool found = false;
        while (std::getline(log_file_, line)) {
            if (line.rfind(target_prefix, 0) == 0) {
                size_t colon_pos = line.find(" : ");
                if (colon_pos != std::string::npos) {
                    std::string message_part = line.substr(colon_pos + 3);
                    std::stringstream ss(message_part);

                    if constexpr (std::is_same_v<T, std::string>) {
                        std::getline(ss, value_from_file);
                    }
                    else {
                        ss >> value_from_file;
                    }

                    found = true;
                    break;
                }
            }
            else if (found) {
                break;
            }
        }

        if (log_file_.is_open()) {
            log_file_.close();
        }
        log_file_.open(filename_, std::ios::out | std::ios::app);
        if (!log_file_.is_open()) {
            throw std::runtime_error("Failed to reopen log file for appending after read: " + filename_);
        }

        if (!found) {
            throw std::runtime_error("Log entry with index " + std::to_string(main_index) + " not found in file.");
        }
        return value_from_file;
    }

    /**
     * @brief Adds a log entry to the in-memory LRU cache used by `FileCache` mode.
     * @details This function implements the Most Recently Used (MRU) logic. If the entry already exists,
     * it is moved to the front of the queue. If the cache exceeds its configured size,
     * the Least Recently Used (LRU) entry is removed from the back.
     * @tparam T The type of the log entry.
     * @param index The logical index of the log entry.
     * @param value The value of the log entry to add.
     */
    template <class T>
    void Log<T>::AddToCache(unsigned int index, const T& value) {
        for (auto it = file_cache_queue_.begin(); it != file_cache_queue_.end(); ++it) {
            if (it->first == index) {
                file_cache_queue_.erase(it);
                break;
            }
        }

        file_cache_queue_.push_front({ index, value });

        while (file_cache_queue_.size() > static_cast<size_t>(file_cache_size_)) {
            file_cache_queue_.pop_back();
        }
    }

    /**
     * @brief Attempts to retrieve a log entry from the in-memory cache in a constant context.
     * @details This function allows peeking into the cache without altering the LRU/MRU order.
     * @tparam T The type of the log entry.
     * @param index The logical index of the log entry to retrieve.
     * @param out_value A reference where the retrieved value will be stored if found.
     * @return True if the entry was found in the cache, false otherwise.
     */
    template <class T>
    bool Log<T>::GetFromCache(unsigned int index, T& out_value) const {
        for (auto it = file_cache_queue_.begin(); it != file_cache_queue_.end(); ++it) {
            if (it->first == index) {
                out_value = it->second;
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Attempts to retrieve a log entry from the in-memory cache, updating its LRU/MRU status.
     * @details If the entry is found, it is moved to the front of the `file_cache_queue_` to mark it as
     * most recently used.
     * @tparam T The type of the log entry.
     * @param index The logical index of the log entry to retrieve.
     * @param out_value A reference where the retrieved value will be stored if found.
     * @return True if the entry was found in the cache, false otherwise.
     */
    template <class T>
    bool Log<T>::GetFromCacheMutable(unsigned int index, T& out_value) {
        for (auto it = file_cache_queue_.begin(); it != file_cache_queue_.end(); ++it) {
            if (it->first == index) {
                out_value = it->second;
                file_cache_queue_.erase(it);
                file_cache_queue_.push_front({ index, out_value });
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Updates the internal `last_accessed_log_index_` to reflect the most recently interacted log entry.
     * @tparam T The type of the log entry.
     * @param index The logical index of the log entry that was just accessed.
     */
    template <class T>
    void Lstream::Log<T>::UpdateLastLogIndex(unsigned int index) const {
        this->last_accessed_log_index_ = index;
    }

    /**
     * @brief Retrieves the current system time.
     * @tparam T The type of the log entry.
     * @return A string representing the current time, formatted as "HH:MM:SS AM/PM".
     */
    template <class T>
    std::string Lstream::Log<T>::GetTimeL() const {
        return GetTimeL(std::chrono::system_clock::now());
    }

    /**
     * @brief Formats a given system time point into a string.
     * @tparam T The type of the log entry.
     * @param tp The `std::chrono::system_clock::time_point` to format.
     * @return A string representing the given time point, formatted as "HH:MM:SS AM/PM".
     */
    template <class T>
    std::string Lstream::Log<T>::GetTimeL(std::chrono::system_clock::time_point tp) const
    {
        std::time_t in_time_t = std::chrono::system_clock::to_time_t(tp);

        std::tm tm{};

    #ifdef _WIN32
        localtime_s(&tm, &in_time_t);
    #else
        localtime_r(&in_time_t, &tm);
    #endif

        std::stringstream ss;
        ss << std::put_time(&tm, "%I:%M:%S %p");

        return ss.str();
    }

    /**
     * @brief Calculates and formats the time elapsed since the `last_log_timestamp_`.
     * @details This function computes the duration between the current time and the last reference timestamp.
     * The `last_log_timestamp_` is then updated to the current time.
     * The output format is milliseconds ("MS") for durations less than a second, and seconds ("S") otherwise.
     * @tparam T The type of the log entry.
     * @return A string representing the time interval (e.g., "50 MS", "3 S").
     */
    template <class T>
    std::string Lstream::Log<T>::GetFormattedTimeInterval() {
        auto current_time = std::chrono::high_resolution_clock::now();
        if (last_log_timestamp_ == std::chrono::high_resolution_clock::time_point()) {
            last_log_timestamp_ = current_time;
            return "0 MS";
        }

        auto duration = current_time - last_log_timestamp_;
        last_log_timestamp_ = current_time;

        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        std::stringstream ss;
        if (ms < 1000 && ms >= 0) {
            ss << ms << " MS";
        }
        else {
            long long s = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
            ss << s << " S";
        }
        return ss.str();
    }

    /**
     * @brief Constructor for the `Log` class.
     * @tparam T The type of the log entry.
     * @param sz The initial size for the in-memory log storage (`Svalue_` and `Occupied_status_`).
     * @param mode The desired logging mode (Cache, File, or FileCache). Defaults to `LogMode::Cache`.
     * @param fname The filename for file-based logging modes. Defaults to "application_log.txt".
     * @param cache_sz The size of the in-memory LRU cache for `FileCache` mode. Defaults to 10.
     * @throw std::invalid_argument if `sz` or `cache_sz` is negative.
     * @throw std::runtime_error if the log file cannot be opened in `File` or `FileCache` mode.
     */
    template <class T>
    Lstream::Log<T>::Log(int sz, LogMode mode, const std::string& fname, int cache_sz) :
        last_accessed_log_index_(0),
        current_last_occupied_log_index_(0),
        last_log_timestamp_(std::chrono::high_resolution_clock::now()),
        size_(sz),
        mode_(mode),
        filename_(fname),
        file_cache_size_(cache_sz)
    {
        if (sz < 0) {
            throw std::invalid_argument("Log size cannot be negative.");
        }
        if (cache_sz < 0) {
            throw std::invalid_argument("File cache size cannot be negative.");
        }

        Svalue_.resize(static_cast<std::size_t>(sz));
        Occupied_status_.resize(static_cast<std::size_t>(sz), 0);


        if (mode_ == LogMode::File || mode_ == LogMode::FileCache) {
            log_file_.open(filename_, std::ios::out | std::ios::app);
            if (!log_file_.is_open()) {
                throw std::runtime_error("Failed to open log file: " + filename_);
            }
        }
    }

    /**
     * @brief Destructor for the `Log` class.
     * @tparam T The type of the log entry.
     * @details Ensures proper closure of the log file stream if it is open, preventing resource leaks.
     */
    template <class T>
    Lstream::Log<T>::~Log() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }

    /**
     * @brief Flushes any pending output in the log file buffer to disk.
     * @details This operation is critical for ensuring data durability, especially in `File` and `FileCache` modes.
     * @tparam T The type of the log entry.
     */
    template <class T>
    void Lstream::Log<T>::FlushFileLog() {
        if (mode_ == LogMode::File || mode_ == LogMode::FileCache) {
            if (log_file_.is_open()) {
                log_file_.flush();
            }
        }
    }

    /**
     * @brief Checks if a given index is within the valid bounds of the `Svalue_` vector.
     * @tparam T The type of the log entry.
     * @param index The index to validate.
     * @return True if the index is valid and within bounds, false otherwise.
     */
    template <class T>
    bool Lstream::Log<T>::DoBoundCheck(unsigned int index) const {
        return index < static_cast<unsigned int>(size_);
    }

    /**
     * @brief Sets a log entry at a specific index with a given value and status.
     * @details This is the core method for adding log data. It handles writing to file (if applicable),
     * updating in-memory storage, and managing the file cache. It also calculates timestamps and intervals.
     * @tparam T The type of the log entry.
     * @param value The log entry data.
     * @param index The logical index at which to store the log.
     * @param status_char A character indicating the status of the log event (e.g., '+', '-').
     * @return True if the log was successfully set.
     * @throw std::out_of_range if the provided index is out of bounds for the in-memory storage.
     */
    template <class T>
    bool Lstream::Log<T>::SetLog(const T& value, unsigned int index, char status_char) {
        UpdateLastLogIndex(index);

        std::string time_str = GetTimeL();
        std::string interval_str = GetFormattedTimeInterval();

        if (mode_ == LogMode::File || mode_ == LogMode::FileCache) {
            if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                unsigned int sub_index = 0;
                for (const auto& msg : value) {
                    WriteLogEntryToFile(index, sub_index++, msg, time_str, interval_str, status_char);
                }
            }
            else {
                std::stringstream ss;
                ss << value;
                WriteLogEntryToFile(index, 0, ss.str(), time_str, interval_str, status_char);
            }
        }

        if (mode_ == LogMode::Cache || mode_ == LogMode::FileCache) {
            if (!DoBoundCheck(index)) {
                throw std::out_of_range("[SetLog Function] Input index is out of bounds for memory allocation: " + std::to_string(index));
            }
            Svalue_[index] = value;
            Occupied_status_[index] = 1;
            if (index >= current_last_occupied_log_index_) {
                current_last_occupied_log_index_ = index;
            }
        }

        if (mode_ == LogMode::FileCache) {
            AddToCache(index, value);
        }
        return true;
    }

    /**
     * @brief Sets a new log entry, automatically determining the next available index.
     * @details This overload simplifies adding logs by handling index management and optional auto-resizing.
     * @tparam T The type of the log entry.
     * @param value The data for the log entry.
     * @param status_char A character indicating the status.
     * @param auto_resize If true, the log's in-memory capacity will expand if needed. Defaults to true.
     * @return True if the log entry was successfully added.
     * @throw std::out_of_range if `auto_resize` is false and no capacity is available.
     */
    template <class T>
    bool Lstream::Log<T>::SetLog(const T& value, char status_char, bool auto_resize) {
        unsigned int next_index;

        if (mode_ == LogMode::Cache || mode_ == LogMode::FileCache) {
            next_index = current_last_occupied_log_index_ + 1;
            if (size_ == 0 || (current_last_occupied_log_index_ == 0 && Occupied_status_[0] == 0)) {
                next_index = 0;
            }


            if (!DoBoundCheck(next_index)) {
                if (auto_resize) {
                    int new_sz = std::max(size_ + 1, size_ * 2);
                    std::cout << "[SetLog Auto] Resizing log from " << size_ << " to " << new_sz << std::endl;
                    Resize(new_sz);
                }
                else {
                    throw std::out_of_range(
                        "[SetLog Auto] Log is full and auto_resize is false. Cannot add new entry."
                    );
                }
            }
        }
        else {
            next_index = current_last_occupied_log_index_ + 1;
            if (current_last_occupied_log_index_ == 0 && file_index_offsets_.empty()) {
                next_index = 0;
            }
            current_last_occupied_log_index_ = next_index;
        }
        return SetLog(value, next_index, status_char);
    }

    /**
     * @brief Checks if a log entry at the specified index is considered 'occupied'.
     * @details 'Occupied' means the index holds a valid log entry. The definition of occupied
     * differs based on the `LogMode`.
     * @tparam T The type of the log entry.
     * @param index The logical index to check.
     * @return True if the index is occupied, false otherwise.
     */
    template <class T>
    bool Lstream::Log<T>::IsOccupied(unsigned int index) const {
        UpdateLastLogIndex(index);

        if (mode_ == LogMode::Cache || mode_ == LogMode::FileCache) {
            if (!DoBoundCheck(index)) {
                return false;
            }
            return Occupied_status_[index] == 1;
        }
        return file_index_offsets_.count(index);
    }

    /**
     * @brief Retrieves a log entry by its logical index, allowing modification.
     * @details This function is the primary way to read log data. Its behavior depends on the current `LogMode`:
     * - `Cache`: Direct access to `Svalue_`.
     * - `FileCache`: Prioritizes `file_cache_queue_` (LRU cache); falls back to file read if not present.
     * - `File`: Reads directly from the log file.
     * The result is returned by reference to allow direct modification if needed.
     * @tparam T The type of the log entry.
     * @param index The logical index of the log entry to retrieve.
     * @return A non-constant reference to the log entry.
     * @throw std::out_of_range if the index is out of the bounds of the in-memory storage.
     * @throw std::runtime_error if no log entry is found for the given index (either not occupied or not found in file).
     */
    template <class T>
    T& Lstream::Log<T>::RetriveLog(unsigned int index) {
        UpdateLastLogIndex(index);

        if (mode_ == LogMode::Cache) {
            if (!DoBoundCheck(index)) {
                throw std::out_of_range("[RetriveLog Function] Input index is out of bounds: " + std::to_string(index));
            }
            if (IsOccupied(index)) {
                return Svalue_[index];
            }
            else {
                throw std::runtime_error("[RetriveLog Function] : No log entry found for the given index: " + std::to_string(index));
            }
        }
        else if (mode_ == LogMode::FileCache) {
            T value_from_source;
            if (GetFromCacheMutable(index, value_from_source)) {
            }
            else {
                value_from_source = ReadLogEntryFromFile(index);
                if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                    if (value_from_source.empty()) {
                        throw std::runtime_error("[RetriveLog Function] : No log entry found for the given index: " + std::to_string(index) + " in file.");
                    }
                }
                AddToCache(index, value_from_source);
            }
            unsigned int target_svalue_idx = DoBoundCheck(index) ? index : 0;
            if (!DoBoundCheck(target_svalue_idx)) {
                Resize(1);
                target_svalue_idx = 0;
            }
            Svalue_[target_svalue_idx] = value_from_source;
            Occupied_status_[target_svalue_idx] = 1;
            return Svalue_[target_svalue_idx];

        }
        else {
            T value_from_file = ReadLogEntryFromFile(index);
            if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                if (value_from_file.empty()) {
                    throw std::runtime_error("[RetriveLog Function] : No log entry found for the given index: " + std::to_string(index) + " in file.");
                }
            }
            unsigned int target_svalue_idx = DoBoundCheck(index) ? index : 0;
            if (!DoBoundCheck(target_svalue_idx)) {
                Resize(1);
                target_svalue_idx = 0;
            }
            Svalue_[target_svalue_idx] = value_from_file;
            Occupied_status_[target_svalue_idx] = 1;
            return Svalue_[target_svalue_idx];
        }
    }

    /**
     * @brief Retrieves a log entry by its logical index in a constant context.
     * @details This function provides read-only access to log data. It behaves similarly to the non-const
     * `RetriveLog` but returns a const reference. For `FileCache` mode, the internal cache's
     * LRU/MRU order might still be updated for optimal future access.
     * @tparam T The type of the log entry.
     * @param index The logical index of the log entry to retrieve.
     * @return A constant reference to the log entry.
     * @throw std::out_of_range if the index is out of the bounds of the in-memory storage.
     * @throw std::runtime_error if no log entry is found for the given index.
     */
    template <class T>
    const T& Lstream::Log<T>::RetriveLog(unsigned int index) const {
        UpdateLastLogIndex(index);

        if (mode_ == LogMode::Cache) {
            if (!DoBoundCheck(index)) {
                throw std::out_of_range("[RetriveLog Function (const)] Input index is out of bounds: " + std::to_string(index));
            }
            if (IsOccupied(index)) {
                return Svalue_[index];
            }
            else {
                throw std::runtime_error("[RetriveLog Function (const)] : No log entry found for the given index: " + std::to_string(index));
            }
        }
        else if (mode_ == LogMode::FileCache) {
            T value_from_source;
            if (const_cast<Log<T>*>(this)->GetFromCacheMutable(index, value_from_source)) {
            }
            else {
                value_from_source = ReadLogEntryFromFile(index);
                if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                    if (value_from_source.empty()) {
                        throw std::runtime_error("[RetriveLog Function (const)] : No log entry found for the given index: " + std::to_string(index) + " in file.");
                    }
                }
                const_cast<Log<T>*>(this)->AddToCache(index, value_from_source);
            }
            unsigned int target_svalue_idx = DoBoundCheck(index) ? index : 0;
            if (!DoBoundCheck(target_svalue_idx)) {
                const_cast<Log<T>*>(this)->Resize(1);
                target_svalue_idx = 0;
            }
            const_cast<Log<T>*>(this)->Svalue_[target_svalue_idx] = value_from_source;
            const_cast<Log<T>*>(this)->Occupied_status_[target_svalue_idx] = 1;
            return const_cast<Log<T>*>(this)->Svalue_[target_svalue_idx];

        }
        else {
            T value_from_file = ReadLogEntryFromFile(index);
            if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                if (value_from_file.empty()) {
                    throw std::runtime_error("[RetriveLog Function (const)] : No log entry found for the given index: " + std::to_string(index) + " in file.");
                }
            }
            unsigned int target_svalue_idx = DoBoundCheck(index) ? index : 0;
            if (!DoBoundCheck(target_svalue_idx)) {
                const_cast<Log<T>*>(this)->Resize(1);
                target_svalue_idx = 0;
            }
            const_cast<Log<T>*>(this)->Svalue_[target_svalue_idx] = value_from_file;
            const_cast<Log<T>*>(this)->Occupied_status_[target_svalue_idx] = 1;
            return const_cast<Log<T>*>(this)->Svalue_[target_svalue_idx];
        }
    }

    /**
     * @brief Clears a log entry by marking its index as unoccupied.
     * @details This operation sets the corresponding entry in `Occupied_status_` to `0`.
     * If in `FileCache` mode, the entry is also removed from the in-memory cache deque.
     * Note: This does not remove the entry from the physical log file, only its in-memory status.
     * @tparam T The type of the log entry.
     * @param index The logical index of the log entry to clear.
     */
    template <class T>
    void Lstream::Log<T>::ClearLog(unsigned int index) {
        if (mode_ == LogMode::Cache || mode_ == LogMode::FileCache) {
            if (DoBoundCheck(index)) {
                Occupied_status_[index] = 0;
            }
            if (mode_ == LogMode::FileCache) {
                for (auto it = file_cache_queue_.begin(); it != file_cache_queue_.end(); ++it) {
                    if (it->first == index) {
                        file_cache_queue_.erase(it);
                        break;
                    }
                }
            }
        }
    }

    /**
     * @brief Resizes the internal in-memory storage vectors (`Svalue_` and `Occupied_status_`).
     * @details Existing data is preserved up to the new size. If `new_sz` is smaller, data beyond
     * the new limit is truncated. Internal index trackers (`last_accessed_log_index_`,
     * `current_last_occupied_log_index_`) are adjusted to remain valid within the new bounds.
     * @tparam T The type of the log entry.
     * @param new_sz The new desired capacity for the in-memory storage.
     * @throw std::invalid_argument if `new_sz` is negative.
     */
    template <class T>
    void Lstream::Log<T>::Resize(int new_sz) {
        if (new_sz < 0) {
            throw std::invalid_argument("[Resize Function] New size cannot be negative.");
        }
        if (new_sz == size_) {
            return;
        }

        if (mode_ == LogMode::Cache || mode_ == LogMode::FileCache || mode_ == LogMode::File) {
            std::vector<T> new_Svalue(static_cast<std::size_t>(new_sz));
            std::vector<char> new_Occupied_Status(static_cast<std::size_t>(new_sz), 0);

            int copy_limit = std::min(size_, new_sz);

            for (int i = 0; i < copy_limit; ++i) {
                new_Svalue[i] = std::move(Svalue_[i]);
                new_Occupied_Status[i] = Occupied_status_[i];
            }

            Svalue_ = std::move(new_Svalue);
            Occupied_status_ = std::move(new_Occupied_Status);
            size_ = new_sz;

            if (last_accessed_log_index_ >= static_cast<unsigned int>(size_)) {
                last_accessed_log_index_ = (size_ > 0 ? static_cast<unsigned int>(size_ - 1) : 0);
            }

            if (current_last_occupied_log_index_ >= static_cast<unsigned int>(size_)) {
                if (size_ == 0) {
                    current_last_occupied_log_index_ = 0;
                }
                else {
                    current_last_occupied_log_index_ = static_cast<unsigned int>(size_ - 1);
                    while (current_last_occupied_log_index_ > 0 && Occupied_status_[current_last_occupied_log_index_] == 0) {
                        current_last_occupied_log_index_--;
                    }
                    if (current_last_occupied_log_index_ == 0 && Occupied_status_[0] == 0) {
                        current_last_occupied_log_index_ = 0;
                    }
                }
            }
        }
    }

    /**
     * @brief Retrieves the index of the log entry that was most recently accessed.
     * @tparam T The type of the log entry.
     * @return The unsigned integer index of the last accessed log entry.
     */
    template <class T>
    unsigned int Lstream::Log<T>::GetLastAccessedLogIndex() const {
        return last_accessed_log_index_;
    }

    /**
     * @brief Retrieves the highest logical index that currently holds an active/occupied log entry.
     * @tparam T The type of the log entry.
     * @return The unsigned integer representing the current highest occupied log index.
     */
    template <class T>
    unsigned int Lstream::Log<T>::GetCurrentLastOccupiedLogIndex() const {
        return current_last_occupied_log_index_;
    }

    /**
     * @brief Retrieves the current allocated capacity of the in-memory log storage.
     * @tparam T The type of the log entry.
     * @return The integer size of the internal `Svalue_` vector.
     */
    template <class T>
    int Lstream::Log<T>::GetSize() const {
        return size_;
    }

    /**
     * @brief Sets the internal reference timestamp for calculating time intervals.
     * @details This function allows users to manually set the starting point for time interval
     * measurements, which can be useful for benchmarking or specific logging scenarios.
     * @tparam T The type of the log entry.
     * @param new_timestamp A `std::chrono::high_resolution_clock::time_point` to use as the new reference.
     */
    template <class T>
    void Lstream::Log<T>::SetIntervalReferenceTimestamp(std::chrono::high_resolution_clock::time_point new_timestamp) {
        last_log_timestamp_ = new_timestamp;
    }

    /**
     * @brief Retrieves the internal `high_resolution_clock::time_point` currently used as the interval reference.
     * @tparam T The type of the log entry.
     * @return The `std::chrono::high_resolution_clock::time_point` of the last interval reference.
     */
    template <class T>
    std::chrono::high_resolution_clock::time_point Lstream::Log<T>::GetLastLogIntervalTimestamp() const {
        return last_log_timestamp_;
    }

    /**
     * @brief Retrieves the internal interval reference timestamp, formatted as "HH:MM:SS AM/PM".
     * @tparam T The type of the log entry.
     * @return A string representation of the last interval reference timestamp.
     */
    template <class T>
    std::string Lstream::Log<T>::GetFormattedLastLogIntervalTimestamp() const {
        return GetTimeL(std::chrono::system_clock::time_point(
            std::chrono::duration_cast<std::chrono::system_clock::duration>(last_log_timestamp_.time_since_epoch())
        ));
    }

} // namespace Lstream
