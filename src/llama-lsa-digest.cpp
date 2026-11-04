#include "llama-lsa-map.h"

#include <openssl/evp.h>
#include <array>
#include <fstream>

namespace llama_lsa {
bool sha256_file(const std::string & path, std::string & digest) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    EVP_MD_CTX * ctx = EVP_MD_CTX_new();
    if (!ctx) return false;
    bool ok = EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1;
    std::array<char, 1 << 20> buffer;
    while (ok && in) {
        in.read(buffer.data(), buffer.size());
        const auto n = in.gcount();
        if (n > 0) ok = EVP_DigestUpdate(ctx, buffer.data(), size_t(n)) == 1;
    }
    ok = ok && in.eof();
    unsigned char bytes[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    if (ok) ok = EVP_DigestFinal_ex(ctx, bytes, &length) == 1 && length == 32;
    EVP_MD_CTX_free(ctx);
    if (!ok) return false;
    static constexpr char hex[] = "0123456789abcdef";
    std::string next(64, '0');
    for (size_t i = 0; i < 32; ++i) {
        next[2*i] = hex[bytes[i] >> 4];
        next[2*i + 1] = hex[bytes[i] & 15];
    }
    digest = std::move(next);
    return true;
}
}
