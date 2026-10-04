#pragma once

#include "pax/common.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

namespace pax {

class MappedFileView {
public:
    MappedFileView() : fd_(-1), data_(nullptr), size_(0) {}

    explicit MappedFileView(const std::string& filepath) : fd_(-1), data_(nullptr), size_(0) {
        open(filepath);
    }

    ~MappedFileView() {
        close();
    }

    MappedFileView(const MappedFileView&) = delete;
    MappedFileView& operator=(const MappedFileView&) = delete;

    MappedFileView(MappedFileView&& other) noexcept
        : fd_(other.fd_), data_(other.data_), size_(other.size_) {
        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
    }

    MappedFileView& operator=(MappedFileView&& other) noexcept {
        if (this != &other) {
            close();
            fd_ = other.fd_;
            data_ = other.data_;
            size_ = other.size_;
            other.fd_ = -1;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void open(const std::string& filepath) {
        close();
        fd_ = ::open(filepath.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        struct stat st;
        if (::fstat(fd_, &st) != 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("Failed to stat file: " + filepath);
        }

        size_ = static_cast<size_t>(st.st_size);
        if (size_ == 0) {
            data_ = nullptr;
            return;
        }

        void* ptr = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (ptr == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("Failed to mmap file: " + filepath);
        }
        data_ = static_cast<const uint8_t*>(ptr);
    }

    void close() {
        if (data_ && size_ > 0) {
            ::munmap(const_cast<void*>(static_cast<const void*>(data_)), size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        size_ = 0;
    }

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_open() const { return fd_ >= 0 || size_ == 0; }

private:
    int fd_;
    const uint8_t* data_;
    size_t size_;
};

inline void write_file(const std::string& filepath, const void* data, size_t size) {
    std::ofstream ofs(filepath, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    if (size > 0 && data) {
        ofs.write(static_cast<const char*>(data), size);
        if (!ofs) {
            throw std::runtime_error("Failed to write to file: " + filepath);
        }
    }
}

inline std::vector<uint8_t> read_file(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (size == 0) return {};
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    ifs.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

} // namespace pax
