#pragma once

#include <vector>
#include <stdint.h>
#include <string>
#include <sstream>
#include <fstream>

struct NpyArray {
    std::vector<size_t> shape;
    std::vector<uint8_t> data;
    size_t elem_size = 0;
    std::string dtype;
};

static bool parse_shape(const std::string& header, std::vector<size_t>& shape) {
    auto shape_pos = header.find("shape");
    if (shape_pos == std::string::npos) return false;
    auto left = header.find('(', shape_pos);
    auto right = header.find(')', left);
    if (left == std::string::npos || right == std::string::npos || right <= left + 1) return false;
    std::string dims = header.substr(left + 1, right - left - 1);
    std::stringstream ss(dims);
    shape.clear();
    while (ss) {
        while (ss.peek() == ' ' || ss.peek() == '\t') ss.get();
        if (ss.peek() == ',') { ss.get(); continue; }
        size_t value;
        if (!(ss >> value)) break;
        shape.push_back(value);
        while (ss.peek() == ' ' || ss.peek() == '\t') ss.get();
        if (ss.peek() == ',') ss.get();
    }
    return !shape.empty();
}

static bool load_npy(const std::string& path, NpyArray& out) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs) {
        fprintf(stderr, "无法打开npy文件: %s\n", path.c_str());
        return false;
    }

    char magic[6];
    fs.read(magic, 6);
    if (fs.gcount() != 6 || std::string(magic, 6) != "\x93NUMPY") {
        fprintf(stderr, "npy文件参数错误: %s\n", path.c_str());
        return false;
    }

    unsigned char major = 0, minor = 0;
    fs.read(reinterpret_cast<char*>(&major), 1);
    fs.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        fs.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else {
        fs.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, ' ');
    fs.read(&header[0], header_len);

    if (header.find("'fortran_order': False") == std::string::npos) {
        fprintf(stderr, "仅支持 C-order npy 文件: %s\n", path.c_str());
        return false;
    }

    auto descr_pos = header.find("'descr':");
    if (descr_pos == std::string::npos) {
        fprintf(stderr, "缺少 descr 字段: %s\n", path.c_str());
        return false;
    }
    auto first_quote = header.find('\'', descr_pos + 8);
    auto second_quote = header.find('\'', first_quote + 1);
    if (first_quote == std::string::npos || second_quote == std::string::npos) {
        fprintf(stderr, "解析 descr 失败: %s\n", path.c_str());
        return false;
    }
    std::string descr = header.substr(first_quote + 1, second_quote - first_quote - 1);

    bool is_float32 = descr.find("f4") != std::string::npos;
    bool is_uint8 = descr.find("u1") != std::string::npos;
    if (!is_float32 && !is_uint8) {
        fprintf(stderr, "仅支持 float32 或 uint8 的npy: %s\n", path.c_str());
        return false;
    }

    size_t elem_size = is_float32 ? sizeof(float) : sizeof(uint8_t);
    bool need_swap = (descr[0] == '>' && elem_size > 1);

    std::vector<size_t> shape;
    if (!parse_shape(header, shape)) {
        fprintf(stderr, "解析 shape 失败: %s\n", path.c_str());
        return false;
    }

    size_t total = 1;
    for (size_t dim : shape) total *= dim;
    if (total == 0) {
        fprintf(stderr, "npy 数据元素数量为0: %s\n", path.c_str());
        return false;
    }

    std::vector<uint8_t> raw(total * elem_size);
    fs.read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (static_cast<size_t>(fs.gcount()) != raw.size()) {
        fprintf(stderr, "读取npy数据失败: %s\n", path.c_str());
        return false;
    }

    if (need_swap && elem_size == sizeof(float)) {
        for (size_t i = 0; i < total; ++i) {
            uint8_t* elem = raw.data() + i * elem_size;
            uint8_t t0 = elem[0]; elem[0] = elem[3]; elem[3] = t0;
            uint8_t t1 = elem[1]; elem[1] = elem[2]; elem[2] = t1;
        }
    }

    out.shape = std::move(shape);
    out.data = std::move(raw);
    out.elem_size = elem_size;
    out.dtype = is_float32 ? "float32" : "uint8";
    return true;
}