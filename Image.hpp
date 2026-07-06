#include "Include/stb_image.h"
#include "Include/stb_image_resize2.h"
#include "Include/stb_image_write.h"
#include "Binary.hpp"
#include "UC_ThreadPool.hpp"
// Main Image Processing Header
#pragma once
#define SPNG_USE_LIBDEFLATE
#define SPNG_STATIC      
#include <string>
#include <string_view>
#include <vector>
#include <fstream>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <cstring>
#include <cctype>
#include <thread>
#include <future>
#include <cmath>
#include <stdexcept>
#include <cassert>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <functional>
#include <cstdlib>
#include <turbojpeg.h>
#include <spng.h>

#define LIBDEFLATE_STATIC
#include "Include/libdeflate/libdeflate.h"
#pragma comment(lib, "Include/libdeflate/libdeflate.lib")

// ========================== SIMD & COMPILER INTRINSICS ========================
#if defined(_MSC_VER)
#define RESTRICT __restrict
#define FORCE_INLINE __forceinline
#include <intrin.h>
#include <malloc.h>
#if defined(_M_X64) || defined(_M_IX86)
#define PREFETCH_L1(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#define PREFETCH_L1(addr)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#define FORCE_INLINE inline __attribute__((always_inline))
#include <cpuid.h>
#define PREFETCH_L1(addr) __builtin_prefetch((addr), 0, 3)
#else
#define RESTRICT
#define FORCE_INLINE inline
#define PREFETCH_L1(addr)
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ARCH_X86
#include <immintrin.h>
#if defined(_MSC_VER)
#define COMPILER_HAS_AVX 1
#define COMPILER_HAS_SSE41 1
#else
#if defined(__AVX__)
#define COMPILER_HAS_AVX 1
#else
#define COMPILER_HAS_AVX 0
#endif
#if defined(__SSE4_1__)
#define COMPILER_HAS_SSE41 1
#else
#define COMPILER_HAS_SSE41 0
#endif
#endif
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
#define ARCH_ARM
#include <arm_neon.h>
#endif

// ========================== INTEL IPP HOOKS (HAL) ========================
#ifdef USE_INTEL_IPP
#include <ippi.h>
#endif

enum class BorderMode { Replicate, Reflect, Constant };

class CpuCapabilities {
public:
    static bool hasSSE41() { static bool res = detectSSE41(); return res; }
    static bool hasAVX() { static bool res = detectAVX(); return res; }
    static bool hasAVX2() { static bool res = detectAVX2(); return res; }
    static bool hasAVX512() { static bool res = detectAVX512(); return res; }
    static bool hasFMA() { static bool res = detectFMA(); return res; }
    static bool hasNEON() {
#if defined(ARCH_ARM)
        return true;
#else
        return false;
#endif
    }

private:
#if defined(ARCH_X86)
    static void cpuid(int info[4], int infoType) {
#if defined(_MSC_VER)
        __cpuidex(info, infoType, 0);
#else
        __cpuid_count(infoType, 0, info[0], info[1], info[2], info[3]);
#endif
    }
    static bool detectSSE41() { int info[4]; cpuid(info, 1); return (info[2] & (1 << 19)) != 0; }
    static bool detectAVX() { int info[4]; cpuid(info, 1); return (info[2] & (1 << 28)) != 0; }
    static bool detectFMA() { int info[4]; cpuid(info, 1); return (info[2] & (1 << 12)) != 0; }
    static bool detectAVX2() { int info[4]; cpuid(info, 7); return (info[1] & (1 << 5)) != 0; }
    static bool detectAVX512() { int info[4]; cpuid(info, 7); return (info[1] & (1 << 16)) != 0; }
#else
    static bool detectSSE41() { return false; }
    static bool detectAVX() { return false; }
    static bool detectFMA() { return false; }
    static bool detectAVX2() { return false; }
    static bool detectAVX512() { return false; }
#endif
};

// ========================== ALIGNED MEMORY & FAST COPY ========================
inline void* aligned_malloc_custom(size_t size, size_t alignment) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    // 100% Portable manual alignment fallback
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    void* ptr = std::malloc(size + alignment + sizeof(void*));
    if (!ptr) return nullptr;
    size_t address = reinterpret_cast<size_t>(ptr) + sizeof(void*);
    size_t offset = alignment - (address % alignment);
    void* aligned_ptr = reinterpret_cast<void*>(address + offset);
    *(reinterpret_cast<void**>(aligned_ptr) - 1) = ptr;
    return aligned_ptr;
#endif
}

inline void aligned_free_custom(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    if (ptr) {
        std::free(*(reinterpret_cast<void**>(ptr) - 1));
    }
#endif
}

FORCE_INLINE void fast_memcpy(void* RESTRICT dst, const void* RESTRICT src, size_t size) {
    std::memcpy(dst, src, size);
}

FORCE_INLINE void fast_memmove(void* dst, const void* src, size_t size) {
    std::memmove(dst, src, size);
}

template<typename T, size_t Alignment = 64>
class AlignedVector {
    static_assert(std::is_trivially_copyable_v<T>, "AlignedVector requires trivially copyable types");
    T* _data = nullptr;
    size_t _size = 0;
    size_t _capacity = 0;

public:
    AlignedVector() = default;
    ~AlignedVector() { if (_data) aligned_free_custom(_data); }

    AlignedVector(const AlignedVector& other) {
        if (other._size > 0) {
            resize_uninitialized(other._size);
            fast_memcpy(_data, other._data, _size * sizeof(T));
        }
    }

    AlignedVector& operator=(const AlignedVector& other) {
        if (this != &other) {
            resize_uninitialized(other._size);
            if (_size > 0) {
                fast_memcpy(_data, other._data, _size * sizeof(T));
            }
        }
        return *this;
    }

    AlignedVector(AlignedVector&& other) noexcept : _data(other._data), _size(other._size), _capacity(other._capacity) {
        other._data = nullptr; other._size = 0; other._capacity = 0;
    }

    AlignedVector& operator=(AlignedVector&& other) noexcept {
        if (this != &other) {
            if (_data) aligned_free_custom(_data);
            _data = other._data; _size = other._size; _capacity = other._capacity;
            other._data = nullptr; other._size = 0; other._capacity = 0;
        }
        return *this;
    }

    void resize_uninitialized(size_t new_size) {
        if (new_size > _capacity) {
            size_t new_cap = std::max(new_size, _capacity * 2);
            if (new_cap < 64) new_cap = 64;
            T* new_data = static_cast<T*>(aligned_malloc_custom(new_cap * sizeof(T), Alignment));
            if (!new_data) throw std::bad_alloc();
            if (_data) {
                if (_size > 0) {
                    fast_memcpy(new_data, _data, _size * sizeof(T));
                }
                aligned_free_custom(_data);
            }
            _data = new_data;
            _capacity = new_cap;
        }
        _size = new_size;
    }

    void assign(const T* first, const T* last) {
        size_t count = last - first;
        resize_uninitialized(count);
        if (count > 0) {
            fast_memcpy(_data, first, count * sizeof(T));
        }
    }

    void clear() { _size = 0; }
    bool empty() const { return _size == 0; }
    size_t size() const { return _size; }
    T* data() { return _data; }
    const T* data() const { return _data; }

    T& operator[](size_t index) { return _data[index]; }
    const T& operator[](size_t index) const { return _data[index]; }
};

// ========================== ZERO-COPY ROI VIEWS ========================
struct ImageView {
    const uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    size_t stride = 0;

    ImageView() = default;
    ImageView(const uint8_t* d, int w, int h, int c, size_t s) : data(d), width(w), height(h), channels(c), stride(s) {}

    ImageView subview(int x, int y, int w, int h) const {
        if (x < 0 || y < 0 || x + w > width || y + h > height) throw std::out_of_range("ROI out of bounds");
        return ImageView(data + y * stride + x * channels, w, h, channels, stride);
    }
};

// ========================== LAZY EVALUATION (EXPRESSION TEMPLATES) ========================
template <typename Derived>
struct ImageExpr {
    FORCE_INLINE uint8_t eval(int x, int y, int c) const {
        return static_cast<const Derived*>(this)->eval_impl(x, y, c);
    }
};

struct GrayscaleExpr : public ImageExpr<GrayscaleExpr> {
    const ImageView src;
    GrayscaleExpr(const ImageView& s) : src(s) {}

    FORCE_INLINE uint8_t eval_impl(int x, int y, int c) const {
        if (src.channels == 1) return src.data[y * src.stride + x];
        const uint8_t* p = src.data + y * src.stride + x * src.channels;
        return static_cast<uint8_t>((p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8);
    }
};

// ========================== CORE EXCEPTION FREE CACHES ========================
struct DeflateContext {
    libdeflate_compressor* compressor = nullptr;
    libdeflate_decompressor* decompressor = nullptr;
    DeflateContext() {
        compressor = libdeflate_alloc_compressor(6);
        decompressor = libdeflate_alloc_decompressor();
    }
    ~DeflateContext() {
        if (compressor) libdeflate_free_compressor(compressor);
        if (decompressor) libdeflate_free_decompressor(decompressor);
    }
    DeflateContext(const DeflateContext&) = delete;
    DeflateContext& operator=(const DeflateContext&) = delete;
};

inline DeflateContext& getDeflateContext() {
    thread_local DeflateContext ctx;
    return ctx;
}

struct ImageResizeWorkspace {
    AlignedVector<int> lutX_indices;
    AlignedVector<float> lutX_weights;
    AlignedVector<int> lutY_indices;
    AlignedVector<float> lutY_weights;
    AlignedVector<float> temp;
    AlignedVector<float> rowAcc;
};

inline ImageResizeWorkspace& getSharedThreadLocalWorkspace() {
    thread_local ImageResizeWorkspace ws;
    return ws;
}

// ========================== HIGH PERFORMANCE THREAD POOL ========================
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;

public:
    ThreadPool(size_t threads = std::thread::hardware_concurrency()) {
        if (threads == 0) threads = 1;
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
                });
        }
    }
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }
    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::move(task));
        }
        condition.notify_one();
    }
    size_t threadCount() const { return workers.size(); }
};

inline ThreadPool& getGlobalThreadPool() {
    static ThreadPool pool;
    return pool;
}

// ==============================================================================

enum class ResizeStrategy { Stretch, AspectFit, AspectFill };

class ImageMetadata {
    using MapType = std::unordered_map<std::string, std::string>;
    std::shared_ptr<MapType> items;

    void ensure_unique() {
        if (!items) items = std::make_shared<MapType>();
        else if (items.use_count() > 1) items = std::make_shared<MapType>(*items);
    }
public:
    ImageMetadata() = default;

    std::string read(const std::string& key, const std::string& fallback = "") const {
        if (!items) return fallback;
        auto search = items->find(key);
        if (search != items->end()) return search->second;
        return fallback;
    }

    bool contains(const std::string& key) const { return items && items->count(key) > 0; }

    void insert(const std::string& key, const std::string& value) { ensure_unique(); (*items)[key] = value; }

    void remove(const std::string& key) {
        if (items && items->count(key) > 0) {
            ensure_unique();
            items->erase(key);
        }
    }

    void clear() { if (items) { if (items.use_count() == 1) items->clear(); else items.reset(); } }
    void strip() { clear(); }

    bool empty() const { return !items || items->empty(); }

    std::unordered_map<std::string, std::string> getAll() const {
        if (items) return *items;
        return {};
    }
};

class ImageBinary : public Binary<ImageBinary> {
public:
    using Binary<ImageBinary>::Binary;

    static ImageBinary loadFromFile(std::string_view filePath) {
        std::ifstream file(std::string(filePath), std::ios::binary | std::ios::ate);
        if (!file.is_open()) throw std::runtime_error("Failed to open file");

        std::streamsize size = file.tellg();
        if (size <= 0) return ImageBinary();

        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> buffer;
        buffer.resize(static_cast<size_t>(size));
        if (file.read(reinterpret_cast<char*>(buffer.data()), size)) return ImageBinary(std::move(buffer));
        throw std::runtime_error("Failed reading file");
    }

    void saveToFile(std::string_view filePath) const {
        if (isEmpty()) throw std::runtime_error("Prevented empty structure save");
        std::ofstream file(std::string(filePath), std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Disk lock error");
        file.write(reinterpret_cast<const char*>(getRawBytes()), getSize());
    }

    ImageBinary compress_impl() const {
        if (isEmpty()) return ImageBinary();
        auto& ctx = getDeflateContext();
        size_t rawSize = getSize();
        size_t limitOutBounds = libdeflate_zlib_compress_bound(ctx.compressor, rawSize);
        std::vector<uint8_t> outBuffer;
        outBuffer.resize(limitOutBounds + sizeof(uint64_t));

        uint64_t originalLenRef = static_cast<uint64_t>(rawSize);
        auto write64_le = [](uint8_t* p, uint64_t v) {
            for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xFF);
            };
        write64_le(outBuffer.data(), originalLenRef);

        size_t compressSizes = libdeflate_zlib_compress(ctx.compressor, getRawBytes(), rawSize, outBuffer.data() + sizeof(uint64_t), limitOutBounds);
        if (compressSizes == 0) throw std::runtime_error("Zlib fault");
        outBuffer.resize(compressSizes + sizeof(uint64_t));
        return ImageBinary(std::move(outBuffer));
    }

    ImageBinary decompress_impl() const {
        if (getSize() <= sizeof(uint64_t)) throw std::runtime_error("Structure sizes implies truncated");
        auto& ctx = getDeflateContext();

        auto read64_le = [](const uint8_t* p) -> uint64_t {
            uint64_t v = 0;
            for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (i * 8);
            return v;
            };
        uint64_t expectedRawSize = read64_le(getRawBytes());

        if (expectedRawSize > 1073741824) throw std::length_error("Suspicious size exceed limits");

        std::vector<uint8_t> finalOutBuffer;
        finalOutBuffer.resize(expectedRawSize);
        size_t actuallyTransacted = 0;
        if (libdeflate_zlib_decompress(ctx.decompressor, getRawBytes() + sizeof(uint64_t), getSize() - sizeof(uint64_t),
            finalOutBuffer.data(), expectedRawSize, &actuallyTransacted) != LIBDEFLATE_SUCCESS) {
            throw std::runtime_error("Corruption blocks extraction");
        }
        return ImageBinary(std::move(finalOutBuffer));
    }
};

class Image {
public:
    enum class LODLevel { Original = 0, High = 1, Medium = 2, Low = 3, Thumbnail = 4 };
    enum class ResizeFilter { Bilinear, Nearest, Bicubic, Lanczos3, Area };
    enum ParseRule { png, jpg, jpeg, bmp, tga, Unknown };
    enum class MimeType { JPEG, PNG, BMP, GIF, WEBP, TIFF, UNKNOWN };

    struct SecurityLimits {
        static constexpr int MaxWidth = 16384;
        static constexpr int MaxHeight = 16384;
        static constexpr size_t MaxPixels = 268435456; // 256 MP (Decompression Bomb Protection)
    };

private:

    // ========================== HIGH-PERFORMANCE ENCODERS ==========================

    std::vector<uint8_t> encodeTurboJPEG(int quality) const {
        tjhandle tjInstance = tjInitCompress();
        if (!tjInstance) throw std::runtime_error("Failed to initialize TurboJPEG");

        short pixelFormat = TJPF_UNKNOWN;
        short subsamp = TJSAMP_444; // Default to highest color resolution

        if (channels == 1) {
            pixelFormat = TJPF_GRAY;
            subsamp = TJSAMP_GRAY;
        }
        else if (channels == 3) {
            pixelFormat = TJPF_RGB;
            // Smart subsampling: Use 4:4:4 for high quality, 4:2:0 for standard web delivery
            subsamp = (quality >= 90) ? TJSAMP_444 : TJSAMP_420;
        }
        else if (channels == 4) {
            pixelFormat = TJPF_RGBA;
            subsamp = (quality >= 90) ? TJSAMP_444 : TJSAMP_420;
        }
        else {
            tjDestroy(tjInstance);
            throw std::runtime_error("Unsupported channel count for JPEG export");
        }

        unsigned char* jpegBuf = nullptr;
        unsigned long jpegSize = 0;

        // tjCompress2 automatically allocates jpegBuf if passed as nullptr
        if (tjCompress2(tjInstance, rawPixels.data(), width, 0, height, pixelFormat,
            &jpegBuf, &jpegSize, subsamp, quality, 0) < 0) {
            std::string err = tjGetErrorStr2(tjInstance);
            tjDestroy(tjInstance);
            throw std::runtime_error("TurboJPEG compression failed: " + err);
        }

        std::vector<uint8_t> result(jpegBuf, jpegBuf + jpegSize);

        tjFree(jpegBuf);
        tjDestroy(tjInstance);

        return result;
    }

    std::vector<uint8_t> encodeSPNG() const {
        spng_ctx* ctx = spng_ctx_new(SPNG_CTX_ENCODER);
        if (!ctx) throw std::runtime_error("Failed to initialize libspng");

        // Tell libspng to allocate the output buffer internally
        spng_set_option(ctx, SPNG_ENCODE_TO_BUFFER, 1);

        // Zlib compression level (1-9). 
        // Level 4 is the sweet spot for speed vs size. STB uses a very slow default.
        // To this:
        spng_set_option(ctx, SPNG_IMG_COMPRESSION_LEVEL, 4);


        spng_ihdr ihdr = { 0 };
        ihdr.width = width;
        ihdr.height = height;
        ihdr.bit_depth = 8;

        if (channels == 1) ihdr.color_type = SPNG_COLOR_TYPE_GRAYSCALE;
        else if (channels == 2) ihdr.color_type = SPNG_COLOR_TYPE_GRAYSCALE_ALPHA;
        else if (channels == 3) ihdr.color_type = SPNG_COLOR_TYPE_TRUECOLOR;
        else if (channels == 4) ihdr.color_type = SPNG_COLOR_TYPE_TRUECOLOR_ALPHA;
        else {
            spng_ctx_free(ctx);
            throw std::runtime_error("Unsupported channel count for PNG export");
        }

        spng_set_ihdr(ctx, &ihdr);

        // Encode the image
        int ret = spng_encode_image(ctx, rawPixels.data(), rawPixels.size(), SPNG_FMT_PNG, SPNG_ENCODE_FINALIZE);
        if (ret != 0) {
            spng_ctx_free(ctx);
            throw std::runtime_error(std::string("libspng encoding failed: ") + spng_strerror(ret));
        }

        // Retrieve the buffer
        size_t png_size = 0;
        void* png_buf = spng_get_png_buffer(ctx, &png_size, &ret);
        if (png_buf == nullptr || ret != 0) {
            spng_ctx_free(ctx);
            throw std::runtime_error("Failed to retrieve PNG buffer from libspng");
        }

        std::vector<uint8_t> result(static_cast<uint8_t*>(png_buf), static_cast<uint8_t*>(png_buf) + png_size);

        std::free(png_buf); // libspng allocates with standard malloc, so use std::free
        spng_ctx_free(ctx);

        return result;
    }
    AlignedVector<uint8_t, 64> rawPixels;
    int width = 0;
    int height = 0;
    int channels = 0;

    std::map<LODLevel, std::shared_ptr<Image>> lods;
    static constexpr float Math_PI = 3.14159265358979323846f;

    // ========================== METADATA EXTRACTION/INJECTION ==========================

    static uint32_t calculateCRC32(const uint8_t* data, size_t length) {
        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < length; i++) {
            crc ^= data[i];
            for (int j = 0; j < 8; j++) {
                crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
            }
        }
        return ~crc;
    }

    void extractPNGMetadata(const uint8_t* data, size_t size) {
        if (size < 8) return;
        size_t offset = 8;
        auto read32_be = [](const uint8_t* p) -> uint32_t {
            return (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
            };
        while (offset + 12 <= size) {
            uint32_t length = read32_be(&data[offset]);
            if (length > size - offset - 12) break;
            std::string type((char*)&data[offset + 4], 4);
            if (type == "tEXt") {
                const uint8_t* chunk_data = &data[offset + 8];
                size_t null_pos = 0;
                while (null_pos < length && chunk_data[null_pos] != '\0') null_pos++;
                if (null_pos < length) {
                    std::string key((char*)chunk_data, null_pos);
                    std::string value((char*)chunk_data + null_pos + 1, length - null_pos - 1);
                    metadata.insert(key, value);
                }
            }
            offset += 12 + length;
        }
    }

    void extractJPEGMetadata(const uint8_t* data, size_t size) {
        if (size < 2 || data[0] != 0xFF || data[1] != 0xD8) return;
        size_t offset = 2;
        auto read16_be = [](const uint8_t* p) -> uint16_t {
            return (p[0] << 8) | p[1];
            };
        while (offset + 4 <= size) {
            if (data[offset] != 0xFF) break;
            uint8_t marker = data[offset + 1];

            if (marker == 0xFF) { offset++; continue; } // Padding
            if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) { offset += 2; continue; } // Standalone
            if (marker == 0xDA) break; // SOS (Start of Scan)

            uint16_t length = read16_be(&data[offset + 2]);
            if (length < 2 || length > size - offset - 2) break;

            if (marker == 0xFE) { // COM (Comment)
                std::string comment((char*)&data[offset + 4], length - 2);
                auto colon = comment.find(": ");
                if (colon != std::string::npos) {
                    metadata.insert(comment.substr(0, colon), comment.substr(colon + 2));
                }
                else {
                    metadata.insert("Comment", comment);
                }
            }
            offset += 2 + length;
        }
    }

    std::vector<uint8_t> injectPNGMetadata(const std::vector<uint8_t>& buffer) const {
        if (buffer.size() < 8) return buffer;
        std::vector<uint8_t> out;
        out.reserve(buffer.size() + 1024);

        out.insert(out.end(), buffer.begin(), buffer.begin() + 8); // Copy signature

        size_t offset = 8;
        if (offset + 8 <= buffer.size()) {
            uint32_t len = (buffer[offset] << 24) | (buffer[offset + 1] << 16) | (buffer[offset + 2] << 8) | buffer[offset + 3];
            if (len <= buffer.size() - offset - 12) {
                std::string type((char*)&buffer[offset + 4], 4);
                if (type == "IHDR") {
                    size_t ihdr_end = offset + 12 + len;
                    out.insert(out.end(), buffer.begin() + offset, buffer.begin() + ihdr_end);
                    offset = ihdr_end;
                }
            }
        }

        auto write32_be = [](std::vector<uint8_t>& vec, uint32_t val) {
            vec.push_back((val >> 24) & 0xFF); vec.push_back((val >> 16) & 0xFF);
            vec.push_back((val >> 8) & 0xFF); vec.push_back(val & 0xFF);
            };

        for (const auto& [key, value] : metadata.getAll()) {
            std::vector<uint8_t> chunk_data;
            chunk_data.insert(chunk_data.end(), key.begin(), key.end());
            chunk_data.push_back(0);
            chunk_data.insert(chunk_data.end(), value.begin(), value.end());

            write32_be(out, chunk_data.size());
            size_t type_offset = out.size();
            out.push_back('t'); out.push_back('E'); out.push_back('X'); out.push_back('t');
            out.insert(out.end(), chunk_data.begin(), chunk_data.end());

            uint32_t crc = calculateCRC32(&out[type_offset], 4 + chunk_data.size());
            write32_be(out, crc);
        }

        out.insert(out.end(), buffer.begin() + offset, buffer.end());
        return out;
    }

    std::vector<uint8_t> injectJPEGMetadata(const std::vector<uint8_t>& buffer) const {
        if (buffer.size() < 2 || buffer[0] != 0xFF || buffer[1] != 0xD8) return buffer;
        std::vector<uint8_t> out;
        out.reserve(buffer.size() + 1024);

        out.push_back(0xFF); out.push_back(0xD8);

        size_t offset = 2;
        if (offset + 4 <= buffer.size() && buffer[offset] == 0xFF && buffer[offset + 1] == 0xE0) {
            uint16_t len = (buffer[offset + 2] << 8) | buffer[offset + 3];
            if (len >= 2 && len <= buffer.size() - offset - 2) {
                out.insert(out.end(), buffer.begin() + offset, buffer.begin() + offset + 2 + len);
                offset += 2 + len;
            }
        }

        auto write16_be = [](std::vector<uint8_t>& vec, uint16_t val) {
            vec.push_back((val >> 8) & 0xFF); vec.push_back(val & 0xFF);
            };

        for (const auto& [key, value] : metadata.getAll()) {
            std::string comment = key + ": " + value;
            if (comment.size() > 65533) comment.resize(65533);

            out.push_back(0xFF); out.push_back(0xFE);
            write16_be(out, comment.size() + 2);
            out.insert(out.end(), comment.begin(), comment.end());
        }

        out.insert(out.end(), buffer.begin() + offset, buffer.end());
        return out;
    }

    template <typename F>
    static void parallel_blocks(int total_items, ExecutionMode mode, int cost_per_item, F&& func) {
        if (total_items <= 0) return;

        bool use_async = (mode == ExecutionMode::Async);

        if (mode == ExecutionMode::Smart) {
            auto& pool = getGlobalThreadPool();
            int num_threads = static_cast<int>(pool.threadCount());
            if (static_cast<int64_t>(total_items) * cost_per_item > 32768LL * num_threads) {
                use_async = true;
            }
        }

        if (!use_async) {
            func(0, total_items);
            return;
        }

        auto& pool = getGlobalThreadPool();
        int num_threads = static_cast<int>(pool.threadCount());
        if (num_threads <= 1) {
            func(0, total_items);
            return;
        }

        int chunk_size = (total_items + num_threads - 1) / num_threads;
        std::atomic<int> tasks_remaining{ num_threads - 1 };
        std::atomic<bool> all_done{ false };
        std::atomic<bool> has_exception{ false };
        std::exception_ptr exception_ptr = nullptr;
        std::mutex exc_mutex;
        std::condition_variable cv;
        std::mutex cv_m;

        for (int i = 0; i < num_threads - 1; ++i) {
            int s = i * chunk_size;
            int e = std::min(s + chunk_size, total_items);
            if (s < e) {
                pool.enqueue([s, e, &func, &tasks_remaining, &cv, &cv_m, &has_exception, &exception_ptr, &exc_mutex, &all_done]() {
                    try {
                        if (!has_exception.load(std::memory_order_relaxed)) {
                            func(s, e);
                        }
                    }
                    catch (...) {
                        has_exception.store(true, std::memory_order_relaxed);
                        std::lock_guard<std::mutex> lock(exc_mutex);
                        if (!exception_ptr) exception_ptr = std::current_exception();
                    }
                    if (tasks_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        {
                            std::lock_guard<std::mutex> lock(cv_m);
                            cv.notify_one();
                        }
                        all_done.store(true, std::memory_order_release);
                    }
                    });
            }
            else {
                if (tasks_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    {
                        std::lock_guard<std::mutex> lock(cv_m);
                        cv.notify_one();
                    }
                    all_done.store(true, std::memory_order_release);
                }
            }
        }

        int s = (num_threads - 1) * chunk_size;
        int e = std::min(s + chunk_size, total_items);
        if (s < e) {
            try {
                if (!has_exception.load(std::memory_order_relaxed)) {
                    func(s, e);
                }
            }
            catch (...) {
                has_exception.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(exc_mutex);
                if (!exception_ptr) exception_ptr = std::current_exception();
            }
        }

        if (tasks_remaining.load(std::memory_order_acquire) > 0) {
            for (int spin = 0; spin < 2000; ++spin) {
                if (tasks_remaining.load(std::memory_order_acquire) == 0) break;
#if defined(ARCH_X86)
                _mm_pause();
#elif defined(ARCH_ARM)
                __asm__ volatile("yield" ::: "memory");
#endif
            }
            if (tasks_remaining.load(std::memory_order_acquire) > 0) {
                std::unique_lock<std::mutex> lock(cv_m);
                cv.wait(lock, [&tasks_remaining] { return tasks_remaining.load(std::memory_order_acquire) == 0; });
            }
        }

        // Ensure the last worker has completely finished using cv_m and cv
        while (!all_done.load(std::memory_order_acquire)) {
#if defined(ARCH_X86)
            _mm_pause();
#elif defined(ARCH_ARM)
            __asm__ volatile("yield" ::: "memory");
#else
            std::this_thread::yield();
#endif
        }

        if (exception_ptr) {
            std::rethrow_exception(exception_ptr);
        }
    }

    static FORCE_INLINE float getBicubicWeight(float x) {
        x = std::abs(x); float x2 = x * x, x3 = x2 * x;
        float a = -0.5f;
        if (x <= 1.0f) return (a + 2.0f) * x3 - (a + 3.0f) * x2 + 1.0f;
        if (x < 2.0f) return a * x3 - 5.0f * a * x2 + 8.0f * a * x - 4.0f * a;
        return 0.0f;
    }

    static FORCE_INLINE float getLanczosWeight(float x, int a = 3) {
        x = std::abs(x);
        if (x < 1e-5f) return 1.0f;
        if (x >= a) return 0.0f;
        float pix = Math_PI * x;
        return (std::sin(pix) * std::sin(pix / a)) / ((pix * pix) / a);
    }

    // ========================== HAL / IPP DISPATCH ==========================
    static bool tryHALResize(const ImageView& src, int dstW, int dstH, AlignedVector<uint8_t, 64>& dst, ResizeFilter filter) {
#ifdef USE_INTEL_IPP
        if (src.channels == 3 && filter == ResizeFilter::Bilinear) {
            dst.resize_uninitialized(static_cast<size_t>(dstW) * dstH * 3);
            IppiSize srcSize = { src.width, src.height };
            IppiRect srcRect = { 0, 0, src.width, src.height };
            IppiSize dstSize = { dstW, dstH };
            int bufSize = 0;
            ippiResizeGetBufSize(srcRect, dstSize, src.channels, IPPI_INTER_LINEAR, &bufSize);
            AlignedVector<uint8_t> buffer; buffer.resize_uninitialized(bufSize);
            ippiResizeLinear_8u_C3R(src.data, src.stride, dst.data(), dstW * 3, dstSize, ippBorderRepl, 0, buffer.data());
            return true;
        }
#endif
        return false;
    }

    // ========================== HIGH-PERFORMANCE KERNELS ==========================

    static void resizeBilinearInternal(const ImageView& src, int dstW, int dstH, AlignedVector<uint8_t, 64>& dst, ExecutionMode mode) {
        int channels = src.channels;
        dst.resize_uninitialized(static_cast<size_t>(dstW) * dstH * channels);

        const uint8_t* RESTRICT srcP = src.data;
        uint8_t* RESTRICT dstP = dst.data();

        int64_t scale_x = (static_cast<int64_t>(src.width) << 16) / dstW;
        int64_t scale_y = (static_cast<int64_t>(src.height) << 16) / dstH;

        parallel_blocks(dstH, mode, dstW * channels, [&](int yStart, int yEnd) {
            for (int dy = yStart; dy < yEnd; ++dy) {
                int64_t sy_fp = (dy * scale_y) + (scale_y >> 1) - 32768;
                int sy = static_cast<int>(sy_fp >> 16);
                int wy = static_cast<int>((sy_fp & 0xFFFF) >> 8);

                int sy0 = std::clamp(sy, 0, src.height - 1);
                int sy1 = std::clamp(sy + 1, 0, src.height - 1);

                const uint8_t* RESTRICT row0 = srcP + sy0 * src.stride;
                const uint8_t* RESTRICT row1 = srcP + sy1 * src.stride;
                uint8_t* RESTRICT dstRow = dstP + dy * dstW * channels;

                int wy0 = 256 - wy;
                int wy1 = wy;

                for (int dx = 0; dx < dstW; ++dx) {
                    int64_t sx_fp = (dx * scale_x) + (scale_x >> 1) - 32768;
                    int sx = static_cast<int>(sx_fp >> 16);
                    int wx = static_cast<int>((sx_fp & 0xFFFF) >> 8);

                    int sx0 = std::clamp(sx, 0, src.width - 1);
                    int sx1 = std::clamp(sx + 1, 0, src.width - 1);

                    int wx0 = 256 - wx;
                    int wx1 = wx;

                    int srcIdx0 = sx0 * channels;
                    int srcIdx1 = sx1 * channels;
                    int dstIdx = dx * channels;

                    PREFETCH_L1(&row0[srcIdx0 + 32]);
                    PREFETCH_L1(&row1[srcIdx0 + 32]);

                    for (int c = 0; c < channels; ++c) {
                        int val0 = (row0[srcIdx0 + c] * wx0 + row0[srcIdx1 + c] * wx1) >> 8;
                        int val1 = (row1[srcIdx0 + c] * wx0 + row1[srcIdx1 + c] * wx1) >> 8;
                        dstRow[dstIdx + c] = static_cast<uint8_t>((val0 * wy0 + val1 * wy1) >> 8);
                    }
                }
            }
            });
    }

    static void resizeAreaInternal(const ImageView& src, int dstW, int dstH, AlignedVector<uint8_t, 64>& dst, ExecutionMode mode) {
        int channels = src.channels;
        dst.resize_uninitialized(static_cast<size_t>(dstW) * dstH * channels);

        float scale_x = static_cast<float>(src.width) / dstW;
        float scale_y = static_cast<float>(src.height) / dstH;

        parallel_blocks(dstH, mode, dstW * channels * 4, [&](int yStart, int yEnd) {
            for (int dy = yStart; dy < yEnd; ++dy) {
                int sy1 = static_cast<int>(dy * scale_y);
                int sy2 = static_cast<int>((dy + 1) * scale_y);
                sy2 = std::min(sy2, src.height);
                if (sy1 == sy2) sy2 = std::min(sy1 + 1, src.height);

                uint8_t* RESTRICT dstRow = dst.data() + dy * dstW * channels;

                for (int dx = 0; dx < dstW; ++dx) {
                    int sx1 = static_cast<int>(dx * scale_x);
                    int sx2 = static_cast<int>((dx + 1) * scale_x);
                    sx2 = std::min(sx2, src.width);
                    if (sx1 == sx2) sx2 = std::min(sx1 + 1, src.width);

                    int area = (sx2 - sx1) * (sy2 - sy1);
                    if (area <= 0) area = 1;

                    int sum[4] = { 0 };

                    for (int y = sy1; y < sy2; ++y) {
                        const uint8_t* RESTRICT srcRow = src.data + y * src.stride;
                        for (int x = sx1; x < sx2; ++x) {
                            int idx = x * channels;
                            for (int c = 0; c < channels; ++c) {
                                sum[c] += srcRow[idx + c];
                            }
                        }
                    }

                    int dstIdx = dx * channels;
                    for (int c = 0; c < channels; ++c) {
                        dstRow[dstIdx + c] = static_cast<uint8_t>(sum[c] / area);
                    }
                }
            }
            });
    }

    template <int CH>
    static void resizeSeparableImpl(
        const ImageView& src, int dstW, int dstH, int kernelRadius,
        const int* RESTRICT l_idxX, const float* RESTRICT l_weiX,
        const int* RESTRICT l_idxY, const float* RESTRICT l_weiY,
        AlignedVector<uint8_t, 64>& dst, ExecutionMode mode)
    {
        int kernelSize = kernelRadius * 2;
        const uint8_t* RESTRICT srcP = src.data;
        size_t srcStride = src.stride;

        dst.resize_uninitialized(static_cast<size_t>(dstW) * dstH * CH);
        uint8_t* RESTRICT dstP = dst.data();

        const bool useSSE41 = CpuCapabilities::hasSSE41();
        const bool useAVX = CpuCapabilities::hasAVX();
        const bool useFMA = CpuCapabilities::hasFMA();
        const bool useNEON = CpuCapabilities::hasNEON();

        parallel_blocks(dstH, mode, dstW * kernelSize * 2, [&](int yStart, int yEnd) {
            auto& ws = getSharedThreadLocalWorkspace();

            int ySrcMin = src.height;
            int ySrcMax = 0;
            for (int dy = yStart; dy < yEnd; ++dy) {
                for (int k = 0; k < kernelSize; ++k) {
                    int sy = l_idxY[dy * kernelSize + k];
                    if (sy < ySrcMin) ySrcMin = sy;
                    if (sy > ySrcMax) ySrcMax = sy;
                }
            }
            if (ySrcMin > ySrcMax) return;

            int srcRows = ySrcMax - ySrcMin + 1;
            ws.temp.resize_uninitialized(dstW * srcRows * CH);
            ws.rowAcc.resize_uninitialized(dstW * CH);

            float* RESTRICT localTemp = ws.temp.data();

            // --- HORIZONTAL PASS ---
            for (int sy = ySrcMin; sy <= ySrcMax; ++sy) {
                const size_t srcRowBase = sy * srcStride;
                int localYRowBase = (sy - ySrcMin) * dstW * CH;

                for (int x = 0; x < dstW; ++x) {
                    int lutBase = x * kernelSize;
                    float* RESTRICT pout = &localTemp[localYRowBase + x * CH];

                    if constexpr (CH == 4) {
                        bool processed = false;
#if defined(ARCH_X86)
#if COMPILER_HAS_SSE41
                        if (useSSE41) {
                            __m128 sum = _mm_setzero_ps();
                            for (int k = 0; k < kernelSize; ++k) {
                                __m128 w = _mm_set1_ps(l_weiX[lutBase + k]);
                                int32_t px32; std::memcpy(&px32, &srcP[srcRowBase + l_idxX[lutBase + k] * 4], 4);
                                __m128i p32 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(px32));
                                __m128 pf = _mm_cvtepi32_ps(p32);
                                if (useFMA) sum = _mm_fmadd_ps(pf, w, sum);
                                else sum = _mm_add_ps(sum, _mm_mul_ps(pf, w));
                            }
                            _mm_storeu_ps(pout, sum);
                            processed = true;
                        }
#endif
#elif defined(ARCH_ARM)
                        if (useNEON) {
                            float32x4_t sum = vdupq_n_f32(0.0f);
                            for (int k = 0; k < kernelSize; ++k) {
                                float32x4_t w = vdupq_n_f32(l_weiX[lutBase + k]);
                                uint8x8_t p8 = vld1_u8(&srcP[srcRowBase + l_idxX[lutBase + k] * 4]);
                                uint16x8_t p16 = vmovl_u8(p8);
                                uint32x4_t p32 = vmovl_u16(vget_low_u16(p16));
                                float32x4_t pf = vcvtq_f32_u32(p32);
                                sum = vmlaq_f32(sum, pf, w);
                            }
                            vst1q_f32(pout, sum);
                            processed = true;
                        }
#endif
                        if (processed) continue;
                    }

                    float s[CH] = { 0.f };
                    for (int k = 0; k < kernelSize; ++k) {
                        float w = l_weiX[lutBase + k];
                        const uint8_t* RESTRICT p = &srcP[srcRowBase + l_idxX[lutBase + k] * CH];
                        for (int c = 0; c < CH; ++c) s[c] += p[c] * w;
                    }
                    for (int c = 0; c < CH; ++c) pout[c] = s[c];
                }
            }

            // --- VERTICAL PASS ---
            for (int dy = yStart; dy < yEnd; ++dy) {
                int outYRowBase = dy * dstW * CH;
                int lutBase = dy * kernelSize;

                std::memset(ws.rowAcc.data(), 0, dstW * CH * sizeof(float));

                for (int k = 0; k < kernelSize; ++k) {
                    float w = l_weiY[lutBase + k];
                    int sy = l_idxY[lutBase + k];
                    const float* RESTRICT pSrcRow = &localTemp[(sy - ySrcMin) * dstW * CH];

                    int totalElements = dstW * CH;
                    int i = 0;

#if defined(ARCH_X86)
#if COMPILER_HAS_AVX
                    if (useAVX) {
                        __m256 vw = _mm256_set1_ps(w);
                        for (; i <= totalElements - 8; i += 8) {
                            PREFETCH_L1(&pSrcRow[i + 32]);
                            __m256 va = _mm256_loadu_ps(&ws.rowAcc[i]);
                            __m256 vb = _mm256_loadu_ps(&pSrcRow[i]);
                            if (useFMA) va = _mm256_fmadd_ps(vb, vw, va);
                            else va = _mm256_add_ps(va, _mm256_mul_ps(vb, vw));
                            _mm256_storeu_ps(&ws.rowAcc[i], va);
                        }
                    }
                    else
#endif
#if COMPILER_HAS_SSE41
                        if (useSSE41) {
                            __m128 vw = _mm_set1_ps(w);
                            for (; i <= totalElements - 4; i += 4) {
                                PREFETCH_L1(&pSrcRow[i + 16]);
                                __m128 va = _mm_loadu_ps(&ws.rowAcc[i]);
                                __m128 vb = _mm_loadu_ps(&pSrcRow[i]);
                                if (useFMA) va = _mm_fmadd_ps(vb, vw, va);
                                else va = _mm_add_ps(va, _mm_mul_ps(vb, vw));
                                _mm_storeu_ps(&ws.rowAcc[i], va);
                            }
                        }
#endif
#elif defined(ARCH_ARM)
                    if (useNEON) {
                        float32x4_t vw = vdupq_n_f32(w);
                        for (; i <= totalElements - 4; i += 4) {
                            float32x4_t va = vld1q_f32(&ws.rowAcc[i]);
                            float32x4_t vb = vld1q_f32(&pSrcRow[i]);
                            va = vmlaq_f32(va, vb, vw);
                            vst1q_f32(&ws.rowAcc[i], va);
                        }
                    }
#endif
                    for (; i < totalElements; ++i) {
                        ws.rowAcc[i] += pSrcRow[i] * w;
                    }
                }

                uint8_t* RESTRICT pD = &dstP[outYRowBase];
                int totalElements = dstW * CH;
                int i = 0;

#if defined(ARCH_X86)
#if COMPILER_HAS_SSE41
                if (useSSE41) {
                    __m128 half = _mm_set1_ps(0.5f);
                    for (; i <= totalElements - 16; i += 16) {
                        __m128 f0 = _mm_add_ps(_mm_loadu_ps(&ws.rowAcc[i]), half);
                        __m128 f1 = _mm_add_ps(_mm_loadu_ps(&ws.rowAcc[i + 4]), half);
                        __m128 f2 = _mm_add_ps(_mm_loadu_ps(&ws.rowAcc[i + 8]), half);
                        __m128 f3 = _mm_add_ps(_mm_loadu_ps(&ws.rowAcc[i + 12]), half);

                        __m128i i0 = _mm_cvttps_epi32(f0);
                        __m128i i1 = _mm_cvttps_epi32(f1);
                        __m128i i2 = _mm_cvttps_epi32(f2);
                        __m128i i3 = _mm_cvttps_epi32(f3);

                        __m128i p0 = _mm_packs_epi32(i0, i1);
                        __m128i p1 = _mm_packs_epi32(i2, i3);
                        __m128i p2 = _mm_packus_epi16(p0, p1);
                        _mm_storeu_si128((__m128i*) & pD[i], p2);
                    }
                }
#endif
#elif defined(ARCH_ARM)
                if (useNEON) {
                    for (; i <= totalElements - 16; i += 16) {
                        float32x4_t f0 = vld1q_f32(&ws.rowAcc[i]);
                        float32x4_t f1 = vld1q_f32(&ws.rowAcc[i + 4]);
                        float32x4_t f2 = vld1q_f32(&ws.rowAcc[i + 8]);
                        float32x4_t f3 = vld1q_f32(&ws.rowAcc[i + 12]);

                        int32x4_t i0 = vcvtaq_s32_f32(f0);
                        int32x4_t i1 = vcvtaq_s32_f32(f1);
                        int32x4_t i2 = vcvtaq_s32_f32(f2);
                        int32x4_t i3 = vcvtaq_s32_f32(f3);

                        int16x4_t s0 = vqmovn_s32(i0);
                        int16x4_t s1 = vqmovn_s32(i1);
                        int16x4_t s2 = vqmovn_s32(i2);
                        int16x4_t s3 = vqmovn_s32(i3);

                        int16x8_t h0 = vcombine_s16(s0, s1);
                        int16x8_t h1 = vcombine_s16(s2, s3);

                        uint8x8_t b0 = vqmovun_s16(h0);
                        uint8x8_t b1 = vqmovun_s16(h1);

                        uint8x16_t b = vcombine_u8(b0, b1);
                        vst1q_u8(&pD[i], b);
                    }
                }
#endif
                for (; i < totalElements; ++i) {
                    pD[i] = static_cast<uint8_t>(std::clamp(static_cast<int>(ws.rowAcc[i] + 0.5f), 0, 255));
                }
            }
            });
    }

    template <typename WeightFunc>
    static void resizeSeparable(
        const ImageView& src, int dstW, int dstH,
        int kernelRadius, WeightFunc weightFunc, AlignedVector<uint8_t, 64>& dst, ExecutionMode mode)
    {
        int kernelSize = kernelRadius * 2;

        AlignedVector<int> lutX_indices; lutX_indices.resize_uninitialized(dstW * kernelSize);
        AlignedVector<float> lutX_weights; lutX_weights.resize_uninitialized(dstW * kernelSize);
        AlignedVector<int> lutY_indices; lutY_indices.resize_uninitialized(dstH * kernelSize);
        AlignedVector<float> lutY_weights; lutY_weights.resize_uninitialized(dstH * kernelSize);

        auto buildLUT = [&](int srcSize, int dstSize, int* indices, float* weights) {
            float ratio = static_cast<float>(srcSize) / dstSize;
            for (int i = 0; i < dstSize; ++i) {
                float center = (i + 0.5f) * ratio - 0.5f;
                int left = static_cast<int>(std::floor(center)) - kernelRadius + 1;
                float weightSum = 0.0f;
                for (int k = 0; k < kernelSize; ++k) {
                    int srcIdx = std::clamp(left + k, 0, srcSize - 1);
                    float w = weightFunc(center - (left + k));
                    weightSum += w;
                    indices[i * kernelSize + k] = srcIdx;
                    weights[i * kernelSize + k] = w;
                }
                if (weightSum != 0.0f) {
                    for (int k = 0; k < kernelSize; ++k) weights[i * kernelSize + k] /= weightSum;
                }
            }
            };

        buildLUT(src.width, dstW, lutX_indices.data(), lutX_weights.data());
        buildLUT(src.height, dstH, lutY_indices.data(), lutY_weights.data());

        switch (src.channels) {
        case 1: resizeSeparableImpl<1>(src, dstW, dstH, kernelRadius, lutX_indices.data(), lutX_weights.data(), lutY_indices.data(), lutY_weights.data(), dst, mode); break;
        case 2: resizeSeparableImpl<2>(src, dstW, dstH, kernelRadius, lutX_indices.data(), lutX_weights.data(), lutY_indices.data(), lutY_weights.data(), dst, mode); break;
        case 3: resizeSeparableImpl<3>(src, dstW, dstH, kernelRadius, lutX_indices.data(), lutX_weights.data(), lutY_indices.data(), lutY_weights.data(), dst, mode); break;
        case 4: resizeSeparableImpl<4>(src, dstW, dstH, kernelRadius, lutX_indices.data(), lutX_weights.data(), lutY_indices.data(), lutY_weights.data(), dst, mode); break;
        default: throw std::runtime_error("Unsupported channel count");
        }
    }

    static void resizeNearestNeighborInternal(
        const ImageView& src, int dstW, int dstH, AlignedVector<uint8_t, 64>& dst, ExecutionMode mode)
    {
        int channels = src.channels;
        dst.resize_uninitialized(static_cast<size_t>(dstW) * dstH * channels);

        AlignedVector<int> lutX; lutX.resize_uninitialized(dstW);
        AlignedVector<size_t> lutY; lutY.resize_uninitialized(dstH);

        float xRatio = static_cast<float>(src.width) / dstW;
        float yRatio = static_cast<float>(src.height) / dstH;

        for (int x = 0; x < dstW; ++x) lutX.data()[x] = std::clamp(static_cast<int>(x * xRatio), 0, src.width - 1) * channels;
        for (int y = 0; y < dstH; ++y) lutY.data()[y] = std::clamp(static_cast<int>(y * yRatio), 0, src.height - 1) * src.stride;

        const uint8_t* RESTRICT srcP = src.data;
        uint8_t* RESTRICT dstP = dst.data();
        const int* RESTRICT lX = lutX.data();
        const size_t* RESTRICT lY = lutY.data();

        parallel_blocks(dstH, mode, dstW, [&](int yStart, int yEnd) {
            for (int y = yStart; y < yEnd; ++y) {
                uint8_t* RESTRICT pDstRow = dstP + (y * dstW * channels);
                const uint8_t* RESTRICT pSrcBase = srcP + lY[y];

                if (channels == 4) {
                    for (int x = 0; x < dstW; ++x) {
                        std::memcpy(pDstRow + (x << 2), pSrcBase + lX[x], 4);
                    }
                }
                else if (channels == 3) {
                    for (int x = 0; x < dstW; ++x) { int dx = x * 3; int sx = lX[x]; pDstRow[dx] = pSrcBase[sx]; pDstRow[dx + 1] = pSrcBase[sx + 1]; pDstRow[dx + 2] = pSrcBase[sx + 2]; }
                }
                else if (channels == 1) {
                    for (int x = 0; x < dstW; ++x) pDstRow[x] = pSrcBase[lX[x]];
                }
                else {
                    for (int x = 0; x < dstW; ++x) { int sx = lX[x]; int dx = x * channels; for (int c = 0; c < channels; ++c) pDstRow[dx + c] = pSrcBase[sx + c]; }
                }
            }
            });
    }

    static void resizeBicubicInternal(const ImageView& src, int dstW, int dstH, AlignedVector<uint8_t, 64>& dst, ExecutionMode mode) {
        resizeSeparable(src, dstW, dstH, 2, [](float x) { return getBicubicWeight(x); }, dst, mode);
    }

    static void resizeLanczos3Internal(const ImageView& src, int dstW, int dstH, AlignedVector<uint8_t, 64>& dst, ExecutionMode mode) {
        resizeSeparable(src, dstW, dstH, 3, [](float x) { return getLanczosWeight(x, 3); }, dst, mode);
    }

    struct ExportBuffer {
        std::vector<uint8_t> buffer;
        ExportBuffer(size_t expected_size) { buffer.reserve(expected_size); }
        static void stbi_write_callback(void* context, void* data, int size) {
            auto* ctx = static_cast<ExportBuffer*>(context);
            auto* bytes = static_cast<const uint8_t*>(data);
            ctx->buffer.insert(ctx->buffer.end(), bytes, bytes + size);
        }
    };

public:
    ImageMetadata metadata;

    Image() = default;

    Image(const Image& other) { copyFrom(other); }
    Image& operator=(const Image& other) { copyFrom(other); return *this; }

    Image(Image&& other) noexcept : rawPixels(std::move(other.rawPixels)), width(other.width), height(other.height), channels(other.channels), lods(std::move(other.lods)), metadata(std::move(other.metadata)) {
        other.width = 0; other.height = 0; other.channels = 0;
    }

    Image& operator=(Image&& other) noexcept {
        if (this != &other) {
            rawPixels = std::move(other.rawPixels);
            width = other.width; height = other.height; channels = other.channels;
            lods = std::move(other.lods);
            metadata = std::move(other.metadata);
            other.width = 0; other.height = 0; other.channels = 0;
        }
        return *this;
    }

    Image(int w, int h, int c, const std::vector<uint8_t>& pixels) : width(w), height(h), channels(c) {
        rawPixels.assign(pixels.data(), pixels.data() + pixels.size());
    }
    explicit Image(const PrimitiveBinary& bin) { loadFromMemory(bin); }
    explicit Image(const ImageBinary& bin) { loadFromMemory(bin.toPrimitive()); }

    void swap(Image& other) noexcept {
        std::swap(rawPixels, other.rawPixels); std::swap(width, other.width); std::swap(height, other.height);
        std::swap(channels, other.channels); std::swap(lods, other.lods); std::swap(metadata, other.metadata);
    }
    void reset() { rawPixels.clear(); width = height = channels = 0; lods.clear(); metadata.clear(); }

    void copyFrom(const Image& other) {
        if (this == &other) return;
        rawPixels = other.rawPixels; width = other.width; height = other.height; channels = other.channels;
        metadata = other.metadata;
        lods = other.lods;
    }

    void copyDataFrom(const Image& other) {
        if (this == &other) return;
        rawPixels = other.rawPixels; width = other.width; height = other.height; channels = other.channels; metadata = other.metadata; lods.clear();
    }

    [[nodiscard]] inline int getWidth() const { return width; }
    [[nodiscard]] inline int getHeight() const { return height; }
    [[nodiscard]] inline int getChannels() const { return channels; }
    [[nodiscard]] inline bool isEmpty() const { return rawPixels.empty() || width <= 0 || height <= 0; }

    [[nodiscard]] ImageView view() const {
        return ImageView(rawPixels.data(), width, height, channels, static_cast<size_t>(width) * channels);
    }

    [[nodiscard]] ImageView subview(int x, int y, int w, int h) const {
        return view().subview(x, y, w, h);
    }

    [[nodiscard]] inline ImageBinary getRawBytes() const {
        std::vector<uint8_t> temp(rawPixels.data(), rawPixels.data() + rawPixels.size());
        return ImageBinary(std::move(temp));
    }

    // ========================== SECURITY & VALIDATION ==========================

    static MimeType detectMimeType(const uint8_t* data, size_t size) {
        if (size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) return MimeType::JPEG;
        if (size >= 8 && data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47) return MimeType::PNG;
        if (size >= 2 && data[0] == 0x42 && data[1] == 0x4D) return MimeType::BMP;
        if (size >= 4 && data[0] == 'G' && data[1] == 'I' && data[2] == 'F' && data[3] == '8') return MimeType::GIF;
        if (size >= 12 && data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' &&
            data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P') return MimeType::WEBP;
        if (size >= 4 && ((data[0] == 'I' && data[1] == 'I' && data[2] == 0x2A && data[3] == 0x00) ||
            (data[0] == 'M' && data[1] == 'M' && data[2] == 0x00 && data[3] == 0x2A))) return MimeType::TIFF;
        return MimeType::UNKNOWN;
    }

    static int extractExifOrientation(const uint8_t* data, size_t size) {
        if (size < 12 || data[0] != 0xFF || data[1] != 0xD8) return 1;
        size_t offset = 2;
        while (offset + 4 <= size) {
            if (data[offset] != 0xFF) break;
            uint8_t marker = data[offset + 1];

            if (marker == 0xFF) { offset++; continue; }
            if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) { offset += 2; continue; }
            if (marker == 0xDA) break;

            uint16_t length = (data[offset + 2] << 8) | data[offset + 3];
            if (length < 2 || length > size - offset - 2) break;

            if (marker == 0xE1) { // APP1
                if (length >= 14 && std::memcmp(&data[offset + 4], "Exif\0\0", 6) == 0) {
                    const uint8_t* tiff = &data[offset + 10];
                    size_t tiffLength = length - 8;
                    if (tiffLength < 8) break;

                    bool littleEndian = (tiff[0] == 'I' && tiff[1] == 'I');
                    auto read16 = [littleEndian](const uint8_t* p) -> uint16_t {
                        return littleEndian ? (p[0] | (p[1] << 8)) : ((p[0] << 8) | p[1]);
                        };
                    auto read32 = [littleEndian](const uint8_t* p) -> uint32_t {
                        return littleEndian ? (p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24))
                            : ((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
                        };

                    uint32_t ifdOffset = read32(&tiff[4]);
                    if (ifdOffset + 2 > tiffLength) break;

                    uint16_t numEntries = read16(&tiff[ifdOffset]);
                    if (ifdOffset + 2 + numEntries * 12 > tiffLength) break;

                    for (uint16_t i = 0; i < numEntries; ++i) {
                        size_t entryOffset = ifdOffset + 2 + i * 12;
                        uint16_t tag = read16(&tiff[entryOffset]);
                        if (tag == 0x0112) { // Orientation tag
                            return read16(&tiff[entryOffset + 8]);
                        }
                    }
                }
            }
            offset += 2 + length;
        }
        return 1;
    }

    void loadFromFile(std::string_view filePath) { loadFromMemory(ImageBinary::loadFromFile(filePath).toPrimitive()); }

    void loadFromMemory(const PrimitiveBinary& inBinary) {
        if (inBinary.isEmpty()) throw std::runtime_error("Void bounds format blocks restrictions");

        const uint8_t* data = inBinary.getRawBytes();
        size_t size = inBinary.getSize();

        if (size > 0x7FFFFFFF) throw std::runtime_error("File too large for STB");

        // 1. MIME Type Validation
        MimeType mime = detectMimeType(data, size);

        // 2. Extract Metadata (Before STB strips it)
        metadata.clear();
        if (mime == MimeType::PNG) extractPNGMetadata(data, size);
        else if (mime == MimeType::JPEG) extractJPEGMetadata(data, size);

        int w, h, c;
        if (!stbi_info_from_memory(data, static_cast<int>(size), &w, &h, &c)) {
            throw std::runtime_error("Unsupported or unrecognized MIME type (Magic number validation failed)");
        }

        // 3. Decompression Bomb Protection
        if (w > SecurityLimits::MaxWidth || h > SecurityLimits::MaxHeight ||
            static_cast<size_t>(w) * h > SecurityLimits::MaxPixels) {
            throw std::runtime_error("Image exceeds maximum allowed dimensions (Decompression Bomb Protection)");
        }

        // 4. Decode
        unsigned char* outMemoryImage = stbi_load_from_memory(data, static_cast<int>(size), &w, &h, &c, 0);
        if (!outMemoryImage) throw std::runtime_error(std::string("STB parser exceptions ") + stbi_failure_reason());

        width = w; height = h; channels = c;
        size_t A = static_cast<size_t>(w) * h * c;
        rawPixels.assign(outMemoryImage, outMemoryImage + A);
        stbi_image_free(outMemoryImage);

        // 5. EXIF Orientation Handling
        if (mime == MimeType::JPEG) {
            int orientation = extractExifOrientation(data, size);
            if (orientation > 1 && orientation <= 8) {
                applyOrientation(orientation);
            }
        }
    }

    // ========================== IN-PLACE & ZERO-COPY OPERATIONS ==========================

    void applyOrientation(int orientation, ExecutionMode mode = ExecutionMode::Smart) {
        switch (orientation) {
        case 2: flipHorizontal(mode); break;
        case 3: rotate180(mode); break;
        case 4: flipVertical(mode); break;
        case 5: flipHorizontal(mode); rotate90CCW(mode); break;
        case 6: rotate90CW(mode); break;
        case 7: flipHorizontal(mode); rotate90CW(mode); break;
        case 8: rotate90CCW(mode); break;
        default: break;
        }
    }

    void rotate90CW(ExecutionMode mode = ExecutionMode::Smart) {
        if (isEmpty()) return;
        AlignedVector<uint8_t, 64> newPixels;
        newPixels.resize_uninitialized(rawPixels.size());
        int oldW = width, oldH = height, ch = channels;
        const uint8_t* src = rawPixels.data();
        uint8_t* dst = newPixels.data();

        parallel_blocks(oldH, mode, oldW * ch, [&](int yStart, int yEnd) {
            for (int y = yStart; y < yEnd; ++y) {
                for (int x = 0; x < oldW; ++x) {
                    int srcIdx = (y * oldW + x) * ch;
                    int dstIdx = (x * oldH + (oldH - 1 - y)) * ch;
                    for (int c = 0; c < ch; ++c) dst[dstIdx + c] = src[srcIdx + c];
                }
            }
            });
        rawPixels = std::move(newPixels);
        std::swap(width, height);
    }

    void rotate90CCW(ExecutionMode mode = ExecutionMode::Smart) {
        if (isEmpty()) return;
        AlignedVector<uint8_t, 64> newPixels;
        newPixels.resize_uninitialized(rawPixels.size());
        int oldW = width, oldH = height, ch = channels;
        const uint8_t* src = rawPixels.data();
        uint8_t* dst = newPixels.data();

        parallel_blocks(oldH, mode, oldW * ch, [&](int yStart, int yEnd) {
            for (int y = yStart; y < yEnd; ++y) {
                for (int x = 0; x < oldW; ++x) {
                    int srcIdx = (y * oldW + x) * ch;
                    int dstIdx = ((oldW - 1 - x) * oldH + y) * ch;
                    for (int c = 0; c < ch; ++c) dst[dstIdx + c] = src[srcIdx + c];
                }
            }
            });
        rawPixels = std::move(newPixels);
        std::swap(width, height);
    }

    void flipHorizontal(ExecutionMode mode = ExecutionMode::Smart) {
        if (isEmpty()) return;
        int w = width, h = height, ch = channels;
        uint8_t* data = rawPixels.data();
        parallel_blocks(h, mode, w * ch, [&](int yStart, int yEnd) {
            for (int y = yStart; y < yEnd; ++y) {
                uint8_t* row = data + y * w * ch;
                for (int x = 0; x < w / 2; ++x) {
                    for (int c = 0; c < ch; ++c) {
                        std::swap(row[x * ch + c], row[(w - 1 - x) * ch + c]);
                    }
                }
            }
            });
    }

    void flipVertical(ExecutionMode mode = ExecutionMode::Smart) {
        if (isEmpty()) return;
        int w = width, h = height, ch = channels;
        uint8_t* data = rawPixels.data();
        size_t rowSize = w * ch;
        parallel_blocks(h / 2, mode, rowSize, [&](int yStart, int yEnd) {
            AlignedVector<uint8_t, 64> tempRow; tempRow.resize_uninitialized(rowSize);
            for (int y = yStart; y < yEnd; ++y) {
                uint8_t* rowTop = data + y * rowSize;
                uint8_t* rowBottom = data + (h - 1 - y) * rowSize;
                if (rowSize > 0) {
                    fast_memcpy(tempRow.data(), rowTop, rowSize);
                    fast_memcpy(rowTop, rowBottom, rowSize);
                    fast_memcpy(rowBottom, tempRow.data(), rowSize);
                }
            }
            });
    }

    void rotate180(ExecutionMode mode = ExecutionMode::Smart) {
        flipVertical(mode);
        flipHorizontal(mode);
    }

    void applyLUTInPlace(const uint8_t* lut) {
        if (isEmpty()) return;
        size_t total = static_cast<size_t>(width) * height * channels;
        uint8_t* RESTRICT p = rawPixels.data();
        for (size_t i = 0; i < total; ++i) {
            p[i] = lut[p[i]];
        }
    }

    void convertToGrayscaleInPlace() {
        if (isEmpty() || channels == 1) return;
        size_t totalPixels = static_cast<size_t>(width) * height;
        uint8_t* RESTRICT p = rawPixels.data();

        for (size_t i = 0; i < totalPixels; ++i) {
            int idx = i * channels;
            p[i] = static_cast<uint8_t>((p[idx] * 77 + p[idx + 1] * 150 + p[idx + 2] * 29) >> 8);
        }
        channels = 1;
        rawPixels.resize_uninitialized(totalPixels);
    }

    template <typename Expr>
    void evaluateFromExpr(const ImageExpr<Expr>& expr, int w, int h, int c) {
        width = w; height = h; channels = c;
        rawPixels.resize_uninitialized(static_cast<size_t>(w) * h * c);
        uint8_t* RESTRICT p = rawPixels.data();

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int ch = 0; ch < c; ++ch) {
                    *p++ = expr.eval(x, y, ch);
                }
            }
        }
    }

    void computeIntegralImage(AlignedVector<uint32_t>& integralOut) const {
        if (isEmpty()) return;
        integralOut.resize_uninitialized(static_cast<size_t>(width) * height * channels);
        uint32_t* RESTRICT out = integralOut.data();
        const uint8_t* RESTRICT in = rawPixels.data();

        for (int c = 0; c < channels; ++c) out[c] = in[c];
        for (int x = 1; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                out[x * channels + c] = out[(x - 1) * channels + c] + in[x * channels + c];
            }
        }

        for (int y = 1; y < height; ++y) {
            uint32_t rowSum[8] = { 0 };
            int rowBase = y * width * channels;
            int prevRowBase = (y - 1) * width * channels;

            for (int x = 0; x < width; ++x) {
                int idx = rowBase + x * channels;
                int prevIdx = prevRowBase + x * channels;
                for (int c = 0; c < channels; ++c) {
                    rowSum[c] += in[idx + c];
                    out[idx + c] = out[prevIdx + c] + rowSum[c];
                }
            }
        }
    }

    static void resolveScalingGeometryConstraintsLimits(int currentW, int currentH, int destW, int destH, ResizeStrategy s, int& targetWOutReturn, int& targetHOutReturn) {
        if (currentW <= 0 || currentH <= 0 || destW <= 0 || destH <= 0) { targetWOutReturn = currentW; targetHOutReturn = currentH; return; }
        if (s == ResizeStrategy::Stretch) { targetWOutReturn = destW; targetHOutReturn = destH; return; }
        float sw = static_cast<float>(destW) / currentW, sh = static_cast<float>(destH) / currentH;
        float usc = (s == ResizeStrategy::AspectFit) ? std::min(sw, sh) : std::max(sw, sh);
        targetWOutReturn = std::max(1, static_cast<int>(currentW * usc)); targetHOutReturn = std::max(1, static_cast<int>(currentH * usc));
    }

    void resizeInto(Image& dst, int requestedW, int requestedH, ResizeStrategy limitsMechanisms = ResizeStrategy::Stretch, ResizeFilter filter = ResizeFilter::Bilinear, ExecutionMode mode = ExecutionMode::Smart) const {
        resizeViewInto(view(), dst, requestedW, requestedH, limitsMechanisms, filter, mode);
    }

    void resizeViewInto(const ImageView& srcView, Image& dst, int requestedW, int requestedH, ResizeStrategy limitsMechanisms = ResizeStrategy::Stretch, ResizeFilter filter = ResizeFilter::Bilinear, ExecutionMode mode = ExecutionMode::Smart) const {
        if (&dst == this) { Image temp; resizeViewInto(srcView, temp, requestedW, requestedH, limitsMechanisms, filter, mode); dst.swap(temp); return; }
        if (srcView.width <= 0 || srcView.height <= 0) { dst.reset(); return; }

        int rX, rY;
        resolveScalingGeometryConstraintsLimits(srcView.width, srcView.height, requestedW, requestedH, limitsMechanisms, rX, rY);

        if (rX <= 0 || rY <= 0) { dst.reset(); return; }

        if (rX == srcView.width && rY == srcView.height && srcView.stride == static_cast<size_t>(srcView.width) * srcView.channels) {
            dst.width = rX; dst.height = rY; dst.channels = srcView.channels;
            dst.rawPixels.assign(srcView.data, srcView.data + (rX * rY * srcView.channels));
            dst.metadata = metadata; dst.lods.clear();
            return;
        }

        dst.width = rX; dst.height = rY; dst.channels = srcView.channels;
        dst.metadata = metadata; dst.lods.clear();

        if (tryHALResize(srcView, rX, rY, dst.rawPixels, filter)) return;

        if (filter == ResizeFilter::Nearest) resizeNearestNeighborInternal(srcView, rX, rY, dst.rawPixels, mode);
        else if (filter == ResizeFilter::Bilinear) resizeBilinearInternal(srcView, rX, rY, dst.rawPixels, mode);
        else if (filter == ResizeFilter::Area) resizeAreaInternal(srcView, rX, rY, dst.rawPixels, mode);
        else if (filter == ResizeFilter::Bicubic) resizeBicubicInternal(srcView, rX, rY, dst.rawPixels, mode);
        else if (filter == ResizeFilter::Lanczos3) resizeLanczos3Internal(srcView, rX, rY, dst.rawPixels, mode);
    }

    void cropSquareCentersInto(Image& dst, int RequestedDim_Width, int RequestedDim_Height) const {
        if (rawPixels.empty()) { dst.reset(); return; }
        int ValidMax_Width = std::min(RequestedDim_Width, width);
        int ValidMax_Height = std::min(RequestedDim_Height, height);
        if (ValidMax_Width <= 0 || ValidMax_Height <= 0) { dst.reset(); return; }
        if (ValidMax_Width == width && ValidMax_Height == height) { dst.copyDataFrom(*this); return; }

        int bOffsetX = (width - ValidMax_Width) / 2;
        int bOffsetY = (height - ValidMax_Height) / 2;

        if (&dst == this) {
            for (int Iteration = 0; Iteration < ValidMax_Height; ++Iteration) {
                int X = ((Iteration + bOffsetY) * width + bOffsetX) * channels;
                int Y = (Iteration * ValidMax_Width) * channels;
                if (X != Y) {
                    fast_memmove(dst.rawPixels.data() + Y, rawPixels.data() + X, static_cast<size_t>(ValidMax_Width) * channels);
                }
            }
            dst.width = ValidMax_Width;
            dst.height = ValidMax_Height;
            dst.rawPixels.resize_uninitialized(static_cast<size_t>(ValidMax_Width) * ValidMax_Height * channels);
            dst.lods.clear();
            return;
        }

        dst.rawPixels.resize_uninitialized(static_cast<size_t>(ValidMax_Width) * ValidMax_Height * channels);
        dst.width = ValidMax_Width; dst.height = ValidMax_Height; dst.channels = channels;
        dst.metadata = metadata; dst.lods.clear();

        for (int Iteration = 0; Iteration < ValidMax_Height; ++Iteration) {
            int X = ((Iteration + bOffsetY) * width + bOffsetX) * channels;
            int Y = (Iteration * ValidMax_Width) * channels;
            fast_memcpy(dst.rawPixels.data() + Y, rawPixels.data() + X, static_cast<size_t>(ValidMax_Width) * channels);
        }
    }

    void cropToSquareInto(Image& dst, int W) const { cropSquareCentersInto(dst, W, W); }
    void resizeAspectFitInto(Image& dst, int mw, int mh, ResizeFilter filter = ResizeFilter::Bilinear, ExecutionMode mode = ExecutionMode::Smart) const { resizeInto(dst, mw, mh, ResizeStrategy::AspectFit, filter, mode); }

    void resizeAspectFillAndCropInto(Image& dst, int SQlimits, ResizeFilter filter = ResizeFilter::Bilinear, ExecutionMode mode = ExecutionMode::Smart) const {
        if (rawPixels.empty()) { dst.reset(); return; }

        float scale = std::max(static_cast<float>(SQlimits) / width, static_cast<float>(SQlimits) / height);
        int srcCropW = static_cast<int>(std::ceil(SQlimits / scale));
        int srcCropH = static_cast<int>(std::ceil(SQlimits / scale));

        srcCropW = std::min(srcCropW, width);
        srcCropH = std::min(srcCropH, height);

        if (srcCropW <= 0 || srcCropH <= 0) { dst.reset(); return; }

        int bOffsetX = (width - srcCropW) / 2;
        int bOffsetY = (height - srcCropH) / 2;

        ImageView roi = subview(bOffsetX, bOffsetY, srcCropW, srcCropH);
        resizeViewInto(roi, dst, SQlimits, SQlimits, ResizeStrategy::Stretch, filter, mode);
    }

    void generateThumbnail(int size = 256, ExecutionMode mode = ExecutionMode::Smart) {
        if (isEmpty()) return;
        if (!lods[LODLevel::Thumbnail] || lods[LODLevel::Thumbnail].use_count() > 1) {
            lods[LODLevel::Thumbnail] = std::make_shared<Image>();
        }
        resizeAspectFillAndCropInto(*lods[LODLevel::Thumbnail], size, ResizeFilter::Area, mode);
    }

    void generateLODsInto(Image& dst, ExecutionMode mode = ExecutionMode::Smart) const {
        if (&dst != this) dst.copyDataFrom(*this);
        if (dst.rawPixels.empty()) return;
        const std::map<LODLevel, int> LOD = { {LODLevel::High, 1024}, {LODLevel::Medium, 512}, {LODLevel::Low, 256} };
        for (const auto& [tier, maxSize] : LOD) {
            if (std::max(dst.width, dst.height) > maxSize) {
                if (!dst.lods[tier] || dst.lods[tier].use_count() > 1) {
                    dst.lods[tier] = std::make_shared<Image>();
                }
                dst.resizeAspectFitInto(*dst.lods[tier], maxSize, maxSize, ResizeFilter::Area, mode);
            }
        }
        // Automatic Thumbnail Generation
        if (!dst.lods[LODLevel::Thumbnail] || dst.lods[LODLevel::Thumbnail].use_count() > 1) {
            dst.lods[LODLevel::Thumbnail] = std::make_shared<Image>();
        }
        dst.resizeAspectFillAndCropInto(*dst.lods[LODLevel::Thumbnail], 128, ResizeFilter::Area, mode);
    }

    void resize(int requestedW, int requestedH, ResizeStrategy limitsMechanisms = ResizeStrategy::Stretch, ResizeFilter filter = ResizeFilter::Bilinear, ExecutionMode mode = ExecutionMode::Smart) { resizeInto(*this, requestedW, requestedH, limitsMechanisms, filter, mode); }
    void cropSquareCenters(int RequestedDim_Width, int RequestedDim_Height) { cropSquareCentersInto(*this, RequestedDim_Width, RequestedDim_Height); }
    void cropToSquare(int W) { cropToSquareInto(*this, W); }
    void resizeAspectFit(int mw, int mh, ResizeFilter filter = ResizeFilter::Bilinear, ExecutionMode mode = ExecutionMode::Smart) { resizeAspectFitInto(*this, mw, mh, filter, mode); }
    void resizeAspectFillAndCrop(int SQlimits, ResizeFilter filter = ResizeFilter::Bilinear, ExecutionMode mode = ExecutionMode::Smart) { resizeAspectFillAndCropInto(*this, SQlimits, filter, mode); }
    void generateLODs(ExecutionMode mode = ExecutionMode::Smart) { generateLODsInto(*this, mode); }

    [[nodiscard]] const Image& getLOD(LODLevel tier) const {
        if (tier == LODLevel::Original) return *this;
        auto it = lods.find(tier); if (it != lods.end()) return *(it->second);
        for (auto Backup : { LODLevel::High, LODLevel::Medium, LODLevel::Low, LODLevel::Thumbnail }) if (lods.count(Backup)) return *lods.at(Backup);
        return *this;
    }

    [[nodiscard]] ImageBinary exportToMemory(ParseRule FormatOutputMechnacis, int ConfigRulesQuality = 85) const {
        if (rawPixels.empty()) throw std::runtime_error("Attempted writing empty structures.");

        std::vector<uint8_t> finalBuffer;

        switch (FormatOutputMechnacis) {
        case jpg:
        case jpeg: {
            finalBuffer = encodeTurboJPEG(ConfigRulesQuality);
            if (!metadata.empty()) finalBuffer = injectJPEGMetadata(finalBuffer);
            break;
        }
        case png: {
            finalBuffer = encodeSPNG();
            if (!metadata.empty()) finalBuffer = injectPNGMetadata(finalBuffer);
            break;
        }
        case bmp: {
            // 14 bytes (File Header) + 40 bytes (Info Header)
            const size_t header_size = 54;
            const size_t pixel_size = static_cast<size_t>(width) * height * channels;

            // Pre-allocate the entire contiguous block to prevent re-allocations
            finalBuffer.resize(header_size + pixel_size);
            uint8_t* dst_ptr = finalBuffer.data();

            // --- Hyper-optimized BMP Header Generation ---
            // Pack fields directly into memory to ensure 100% alignment safety across architectures

            // File Header (14 bytes)
            dst_ptr[0] = 'B'; dst_ptr[1] = 'M'; // Signature
            *reinterpret_cast<uint32_t*>(dst_ptr + 2) = static_cast<uint32_t>(header_size + pixel_size); // File Size
            *reinterpret_cast<uint32_t*>(dst_ptr + 6) = 0;  // Reserved
            *reinterpret_cast<uint32_t*>(dst_ptr + 10) = static_cast<uint32_t>(header_size); // Pixel Data Offset

            // Info Header (40 bytes)
            *reinterpret_cast<uint32_t*>(dst_ptr + 14) = 40; // Header Size
            *reinterpret_cast<int32_t*>(dst_ptr + 18) = width;
            *reinterpret_cast<int32_t*>(dst_ptr + 22) = -height; // Negative height flips image so top-left is (0,0)
            *reinterpret_cast<uint16_t*>(dst_ptr + 26) = 1; // Planes
            *reinterpret_cast<uint16_t*>(dst_ptr + 28) = static_cast<uint16_t>(channels * 8); // Bits per pixel (24 or 32)
            *reinterpret_cast<uint32_t*>(dst_ptr + 30) = 0; // Compression (0 = BI_RGB uncompressed)
            *reinterpret_cast<uint32_t*>(dst_ptr + 34) = static_cast<uint32_t>(pixel_size);
            *reinterpret_cast<int32_t*>(dst_ptr + 38) = 2835; // XPelsPerMeter (~72 DPI)
            *reinterpret_cast<int32_t*>(dst_ptr + 42) = 2835; // YPelsPerMeter
            *reinterpret_cast<uint32_t*>(dst_ptr + 46) = 0; // Colors used
            *reinterpret_cast<uint32_t*>(dst_ptr + 50) = 0; // Important colors

            // --- Hyper-optimized Pixel Blit ---
            fast_memcpy(dst_ptr + header_size, rawPixels.data(), pixel_size);
            break;
        }
        case tga: {
            constexpr uint8_t header_size = 18;
            const size_t pixel_size = static_cast<size_t>(width) * height * channels;

            finalBuffer.resize(header_size + pixel_size);
            uint8_t* dst_ptr = finalBuffer.data();

            // --- Hyper-optimized TGA Header Generation ---
            dst_ptr[0] = 0;
            dst_ptr[1] = 0;
            dst_ptr[2] = 2; // Uncompressed true-color
            *reinterpret_cast<uint16_t*>(dst_ptr + 3) = 0;
            *reinterpret_cast<uint16_t*>(dst_ptr + 5) = 0;
            dst_ptr[7] = 0;
            *reinterpret_cast<uint16_t*>(dst_ptr + 8) = 0;
            *reinterpret_cast<uint16_t*>(dst_ptr + 10) = 0;
            *reinterpret_cast<uint16_t*>(dst_ptr + 12) = static_cast<uint16_t>(width);
            *reinterpret_cast<uint16_t*>(dst_ptr + 14) = static_cast<uint16_t>(height);
            dst_ptr[16] = static_cast<uint8_t>(channels * 8);
            dst_ptr[17] = 0x20; // Top-left origin descriptor


            fast_memcpy(dst_ptr + header_size, rawPixels.data(), pixel_size);

            break;
        }
        default:
            throw std::invalid_argument("Failed boundaries constraints");
        }
        return ImageBinary(std::move(finalBuffer));
    }

    [[nodiscard]] ParseRule ExtentionResolve(std::string_view ex) const {
        std::string ext = std::string(ex); auto dot_pos = ext.rfind('.');
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        ext = (dot_pos == std::string::npos) ? ext : ext.substr(dot_pos + 1);
        if (ext == "png") return ParseRule::png; else if (ext == "jpg" || ext == "jpeg") return ParseRule::jpeg;
        else if (ext == "bmp") return ParseRule::bmp; else if (ext == "tga") return ParseRule::tga;
        return ParseRule::Unknown;
    }

    void saveToFile(std::string_view path, int Rule = 85) const { exportToMemory(ExtentionResolve(path), Rule).saveToFile(path); }

    [[nodiscard]] static Image fromFile(std::string_view path) { Image t; t.loadFromFile(path); return t; }
    [[nodiscard]] static Image fromMemory(const PrimitiveBinary& bin) { Image t; t.loadFromMemory(bin); return t; }
    [[nodiscard]] static ImageBinary convertFormat(const PrimitiveBinary& bin, ParseRule Rule, int Dim_H = 85) { return fromMemory(bin).exportToMemory(Rule, Dim_H); }
    [[nodiscard]] static ImageBinary resizeToMemory(const PrimitiveBinary& bin, int sz, int vs, ResizeStrategy Strategy, ParseRule ex, int v = 85, ResizeFilter filter = ResizeFilter::Bilinear, ExecutionMode mode = ExecutionMode::Smart) {
        Image t_img = fromMemory(bin); t_img.resize(sz, vs, Strategy, filter, mode); return t_img.exportToMemory(ex, v);
    }
    [[nodiscard]] static ImageBinary downscaleToMemory(const PrimitiveBinary& bin, int Dr, int Map_H, ParseRule ex, int vs = 85, ResizeFilter filter = ResizeFilter::Area, ExecutionMode mode = ExecutionMode::Smart) {
        return resizeToMemory(bin, Dr, Map_H, ResizeStrategy::AspectFit, ex, vs, filter, mode);
    }
    [[nodiscard]] static ImageBinary createThumbnail(const PrimitiveBinary& bin, int height, ParseRule Formats, int v = 85) {
        Image t_img = fromMemory(bin); t_img.cropToSquare(height); return t_img.exportToMemory(Formats, v);
    }
    [[nodiscard]] static ImageBinary compress(const PrimitiveBinary& bin) { return ImageBinary(bin.getRawBytes(), bin.getSize()).compress(); }
    [[nodiscard]] static ImageBinary decompress(const PrimitiveBinary& bin) { return ImageBinary(bin.getRawBytes(), bin.getSize()).decompress(); }
};

class SharedImage {
    std::shared_ptr<Image> img;
public:
    SharedImage() : img(std::make_shared<Image>()) {}
    explicit SharedImage(std::shared_ptr<Image> i) : img(std::move(i)) {}
    explicit SharedImage(Image&& i) : img(std::make_shared<Image>(std::move(i))) {}
    static SharedImage fromFile(std::string_view i) { return SharedImage(std::make_shared<Image>(Image::fromFile(i))); }
    static SharedImage fromMemory(const PrimitiveBinary& i) { return SharedImage(std::make_shared<Image>(Image::fromMemory(i))); }

    SharedImage(const SharedImage&) = default;
    SharedImage& operator=(const SharedImage&) = default;
    SharedImage(SharedImage&&) noexcept = default;
    SharedImage& operator=(SharedImage&&) noexcept = default;
    Image* operator->() { return img.get(); }
    const Image* operator->() const { return img.get(); }
    Image& operator*() { return *img; }
    const Image& operator*() const { return *img; }
    [[nodiscard]] explicit operator bool() const noexcept { return img != nullptr && !img->isEmpty(); }
    [[nodiscard]] Image* get() const noexcept { return img.get(); }
};