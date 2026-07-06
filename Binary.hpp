// Binary.hpp
#pragma once
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

enum class ExecutionMode {
    Sync,      // Always execute synchronously on the calling thread.
    Async,     // Always execute asynchronously via the thread pool.
    Smart      // Automatically decide based on workload and overhead.
};

class PrimitiveBinary {
private:
    const uint8_t* _rawBytes = nullptr;
    size_t _size = 0;

public:
    PrimitiveBinary() = default;

    PrimitiveBinary(const uint8_t* dataPointer, size_t dataSize)
        : _rawBytes(dataPointer), _size(dataSize) {
    }

    PrimitiveBinary(const char* charPointer, size_t dataSize)
        : _rawBytes(reinterpret_cast<const uint8_t*>(charPointer)), _size(dataSize) {
    }

    PrimitiveBinary(const std::vector<uint8_t>& buf)
        : _rawBytes(buf.data()), _size(buf.size()) {
    }

    PrimitiveBinary(const PrimitiveBinary&) = default;
    PrimitiveBinary& operator=(const PrimitiveBinary&) = default;

    PrimitiveBinary(PrimitiveBinary&& other) noexcept : _rawBytes(other._rawBytes), _size(other._size) {
        other._rawBytes = nullptr;
        other._size = 0;
    }

    PrimitiveBinary& operator=(PrimitiveBinary&& other) noexcept {
        if (this != &other) {
            _rawBytes = other._rawBytes;
            _size = other._size;
            other._rawBytes = nullptr;
            other._size = 0;
        }
        return *this;
    }

    [[nodiscard]] inline const uint8_t* getRawBytes() const noexcept { return _rawBytes; }
    [[nodiscard]] inline size_t getSize() const noexcept { return _size; }
    [[nodiscard]] inline bool isEmpty() const noexcept { return _size == 0 || _rawBytes == nullptr; }

    [[nodiscard]] inline uint8_t getByte(size_t index) const {
        if (index >= _size) throw std::out_of_range("PrimitiveBinary index out of bounds.");
        return _rawBytes[index];
    }

    [[nodiscard]] inline uint8_t operator[](size_t index) const noexcept {
        if (index >= _size || _rawBytes == nullptr) return 0;
        return _rawBytes[index];
    }

    [[nodiscard]] inline PrimitiveBinary slice(size_t offset, size_t length) const {
        if (offset > _size || length > _size - offset) {
            throw std::out_of_range("Requested slice exceeds data boundaries.");
        }
        return PrimitiveBinary(_rawBytes + offset, length);
    }
};

template<typename Derived>
class Binary {
protected:
    std::vector<uint8_t> data;

public:
    Binary() = default;
    explicit Binary(size_t size) : data(size, 0) {}
    Binary(const uint8_t* byteBuffer, size_t size) : data(byteBuffer, byteBuffer + size) {}
    explicit Binary(std::vector<uint8_t> buffer) : data(std::move(buffer)) {}

    Binary(const Binary& other) = default;
    Binary& operator=(const Binary& other) = default;

    Binary(Binary&& other) noexcept : data(std::move(other.data)) {}
    Binary& operator=(Binary&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
        }
        return *this;
    }

    [[nodiscard]] inline const uint8_t* getRawBytes() const { return data.data(); }
    [[nodiscard]] inline uint8_t* getRawBytes() { return data.data(); }
    [[nodiscard]] inline size_t getSize() const { return data.size(); }
    [[nodiscard]] inline bool isEmpty() const { return data.empty(); }

    void inline clear() { data.clear(); }
    void inline resize(size_t size) { data.resize(size); }

    [[nodiscard]] inline PrimitiveBinary toPrimitive() const noexcept {
        return PrimitiveBinary(data.data(), data.size());
    }

    Derived compress() const {
        return static_cast<const Derived&>(*this).compress_impl();
    }

    Derived decompress() const {
        return static_cast<const Derived&>(*this).decompress_impl();
    }
};