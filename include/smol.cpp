#ifndef SMOL_HPP
#define SMOL_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "llama.h"

namespace py = pybind11;

namespace smol {

// Arithmetic coding constants
const uint8_t PRECISION = 32;
const uint64_t MAX_RANGE = (1ULL << 32) - 1;
const uint64_t HALF = 1ULL << 31;
const uint64_t QUARTER = 1ULL << 30;
const uint64_t THREE_QUARTERS = 3ULL << 30;
const size_t WINDOW_SIZE = 64;

// Helper functions for bit manipulation
std::vector<uint8_t> bits_to_bytes(const std::vector<int>& bits) {
    // Pad to byte width
    size_t padding = (8 - bits.size() % 8) % 8;
    std::vector<int> padded_bits = bits;
    padded_bits.insert(padded_bits.end(), padding, 0);

    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < padded_bits.size(); i += 8) {
        uint8_t byte = 0;
        for (size_t j = 0; j < 8; j++) {
            byte = (byte << 1) | padded_bits[i + j];
        }
        bytes.push_back(byte);
    }
    return bytes;
}

std::vector<int> bytes_to_bits(const std::vector<uint8_t>& data) {
    std::vector<int> bits;
    for (uint8_t byte : data) {
        for (int i = 7; i >= 0; i--) {
            bits.push_back((byte >> i) & 1);
        }
    }
    return bits;
}

class Compressor {
private:
    llama_model* model;
    llama_context* ctx;
    std::string model_path;
    size_t kv_cache_size;  // Track how many tokens are in KV cache

    // Helper: Get probability distribution for next token
    std::vector<double> get_token_probs(const std::vector<int32_t>& context_tokens) {
        const llama_vocab* vocab = llama_model_get_vocab(model);
        int32_t n_vocab = llama_vocab_n_tokens(vocab);

        // Check if we hit a window boundary - if so, reset completely
        bool at_window_boundary = (context_tokens.size() > 0 && context_tokens.size() % WINDOW_SIZE == 0);

        // Check if we need to rebuild context
        // Always rebuild if context is empty (to evaluate BOS)
        // Or if cache size doesn't match context size
        // Or if we hit a window boundary
        bool need_rebuild = context_tokens.empty() ||
                           (kv_cache_size != context_tokens.size()) ||
                           at_window_boundary;

        if (need_rebuild) {
            // Note: We don't recreate context here since this is called in a loop
            // The context was freshly created at the start of compress/decompress
            // Just reset the KV cache tracking
            kv_cache_size = 0;

            if (context_tokens.empty() || at_window_boundary) {
                // Evaluate BOS to get first token probs (or reset at boundary)
                llama_token bos = llama_vocab_bos(vocab);
                llama_decode(ctx, llama_batch_get_one(&bos, 1));
            } else {
                // Evaluate all context tokens
                for (size_t i = 0; i < context_tokens.size(); i++) {
                    llama_token token = context_tokens[i];
                    llama_decode(ctx, llama_batch_get_one(&token, 1));
                }
                kv_cache_size = context_tokens.size();
            }
        }

        // Get logits for the last position (next token prediction)
        float* logits = llama_get_logits(ctx);

        // Convert to probabilities with numerically stable softmax
        std::vector<double> probs(n_vocab);
        double max_logit = *std::max_element(logits, logits + n_vocab);

        double sum_exp = 0.0;
        for (int32_t i = 0; i < n_vocab; i++) {
            probs[i] = std::exp(static_cast<double>(logits[i]) - max_logit);
            sum_exp += probs[i];
        }

        for (int32_t i = 0; i < n_vocab; i++) {
            probs[i] /= sum_exp;
        }

        return probs;
    }

public:
    // Constructor - loads the model
    Compressor(const std::string& path) : model_path(path), ctx(nullptr), kv_cache_size(0) {
        llama_model_params params = llama_model_default_params();
        params.vocab_only = false;

        model = llama_load_model_from_file(path.c_str(), params);
        if (!model) {
            throw std::runtime_error("Failed to load model: " + path);
        }

        // Create context for inference
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = WINDOW_SIZE * 2;  // Allow some buffer
        ctx_params.n_batch = 1;

        ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            llama_free_model(model);
            throw std::runtime_error("Failed to create context");
        }
    }

    // Destructor - frees the model and context
    ~Compressor() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model) {
            llama_free_model(model);
            model = nullptr;
        }
    }

    // Delete copy constructor and assignment operator to prevent double-free
    Compressor(const Compressor&) = delete;
    Compressor& operator=(const Compressor&) = delete;

    // Compress function - takes string, returns bit stream
    std::vector<uint8_t> compress(const std::string& data) {
        if (!model || !ctx) {
            throw std::runtime_error("Model not loaded");
        }

        // Tokenize input (don't add BOS - we handle it in get_token_probs)
        const llama_vocab* vocab = llama_model_get_vocab(model);
        std::vector<int32_t> tokens(data.size() + 16);  // Allocate enough space
        int32_t n_tokens = llama_tokenize(
            vocab,
            data.c_str(),
            data.size(),
            tokens.data(),
            tokens.size(),
            false,  // add_bos
            true    // special
        );

        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, data.c_str(), data.size(), tokens.data(), tokens.size(), false, true);
        }
        tokens.resize(n_tokens);

        // Recreate context to ensure clean state
        if (ctx) {
            llama_free(ctx);
        }
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = WINDOW_SIZE * 2;
        ctx_params.n_batch = 1;
        ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            throw std::runtime_error("Failed to recreate context");
        }
        kv_cache_size = 0;

        // Arithmetic coding
        uint64_t lo = 0;
        uint64_t hi = MAX_RANGE;
        std::vector<int> output_bits;
        int underflow_count = 0;

        for (size_t i = 0; i < tokens.size(); i++) {
            // Get context up to current position (tokens 0 to i-1)
            std::vector<int32_t> context(tokens.begin(), tokens.begin() + i);
            std::vector<double> probs = get_token_probs(context);
            // After get_token_probs, kv_cache_size = i (we have context of i tokens)

            int32_t token = tokens[i];
            const llama_vocab* vocab = llama_model_get_vocab(model);
            int32_t n_vocab = llama_vocab_n_tokens(vocab);

            // Sort tokens by probability
            std::vector<int32_t> sorted_tokens(n_vocab);
            for (int32_t j = 0; j < n_vocab; j++) sorted_tokens[j] = j;
            std::sort(sorted_tokens.begin(), sorted_tokens.end(),
                     [&probs](int32_t a, int32_t b) { return probs[a] > probs[b]; });

            // Find rank and calculate cumulative probability
            int32_t rank = 0;
            for (; rank < n_vocab; rank++) {
                if (sorted_tokens[rank] == token) break;
            }

            double prob_before = 0.0;
            for (int32_t j = 0; j < rank; j++) {
                prob_before += probs[sorted_tokens[j]];
            }
            double token_prob = probs[token];

            // Update interval (using 64-bit to detect overflow)
            uint64_t width = hi - lo;
            lo = lo + static_cast<uint64_t>(prob_before * width);
            hi = lo + std::max(1ULL, static_cast<uint64_t>(token_prob * width));

            // Add this token to KV cache for next iteration
            llama_decode(ctx, llama_batch_get_one(&token, 1));
            kv_cache_size++;

            // Renormalize
            while (true) {
                if (hi <= HALF) {
                    output_bits.push_back(0);
                    for (int j = 0; j < underflow_count; j++) output_bits.push_back(1);
                    underflow_count = 0;
                    lo = lo << 1;
                    hi = hi << 1;
                } else if (lo >= HALF) {
                    output_bits.push_back(1);
                    for (int j = 0; j < underflow_count; j++) output_bits.push_back(0);
                    underflow_count = 0;
                    lo = (lo - HALF) << 1;
                    hi = (hi - HALF) << 1;
                } else if (lo >= QUARTER && hi <= THREE_QUARTERS) {
                    underflow_count++;
                    lo = (lo - QUARTER) << 1;
                    hi = (hi - QUARTER) << 1;
                } else {
                    break;
                }
            }
        }

        // Flush remaining bits
        underflow_count++;
        if (lo < QUARTER) {
            output_bits.push_back(0);
            for (int j = 0; j < underflow_count; j++) output_bits.push_back(1);
        } else {
            output_bits.push_back(1);
            for (int j = 0; j < underflow_count; j++) output_bits.push_back(0);
        }

        // Convert bits to bytes
        std::vector<uint8_t> compressed_data = bits_to_bytes(output_bits);

        // Prepend token count (big-endian 4 bytes)
        std::vector<uint8_t> result(4 + compressed_data.size());
        result[0] = (tokens.size() >> 24) & 0xFF;
        result[1] = (tokens.size() >> 16) & 0xFF;
        result[2] = (tokens.size() >> 8) & 0xFF;
        result[3] = tokens.size() & 0xFF;
        std::copy(compressed_data.begin(), compressed_data.end(), result.begin() + 4);

        return result;
    }

    // Decompress function - takes bit stream, returns string or throws
    std::string decompress(const std::vector<uint8_t>& bitstream) {
        if (!model || !ctx) {
            throw std::runtime_error("Model not loaded");
        }

        if (bitstream.size() < 4) {
            throw std::runtime_error("Invalid compressed data: too short");
        }

        // Read token count (big-endian 4 bytes)
        uint32_t num_tokens = (static_cast<uint32_t>(bitstream[0]) << 24) |
                              (static_cast<uint32_t>(bitstream[1]) << 16) |
                              (static_cast<uint32_t>(bitstream[2]) << 8) |
                              static_cast<uint32_t>(bitstream[3]);

        // Extract compressed data
        std::vector<uint8_t> compressed_data(bitstream.begin() + 4, bitstream.end());
        std::vector<int> bits = bytes_to_bits(compressed_data);

        // Recreate context to ensure clean state
        if (ctx) {
            llama_free(ctx);
        }
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = WINDOW_SIZE * 2;
        ctx_params.n_batch = 1;
        ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            throw std::runtime_error("Failed to recreate context");
        }
        kv_cache_size = 0;

        // Read initial value from bitstream (32-bit)
        uint32_t value = 0;
        size_t bit_index = 0;
        for (uint8_t i = 0; i < PRECISION; i++) {
            value = value << 1;
            if (bit_index < bits.size()) {
                value = value | bits[bit_index];
                bit_index++;
            }
        }

        const llama_vocab* vocab = llama_model_get_vocab(model);
        int32_t n_vocab = llama_vocab_n_tokens(vocab);

        uint64_t lo = 0;
        uint64_t hi = MAX_RANGE;
        std::vector<int32_t> decompressed_tokens;

        for (uint32_t i = 0; i < num_tokens; i++) {
            std::vector<double> probs = get_token_probs(decompressed_tokens);
            // After get_token_probs, kv_cache_size = decompressed_tokens.size()

            // Sort tokens by probability
            std::vector<int32_t> sorted_tokens(n_vocab);
            for (int32_t j = 0; j < n_vocab; j++) sorted_tokens[j] = j;
            std::sort(sorted_tokens.begin(), sorted_tokens.end(),
                     [&probs](int32_t a, int32_t b) { return probs[a] > probs[b]; });

            // Find token based on value position
            uint64_t width = hi - lo;
            double position = static_cast<double>(value - lo) / width;

            // Clamp position
            if (position < 0) position = 0;
            if (position >= 1.0) position = 0.9999999;

            // Build CDF and find token
            std::vector<double> cdf(n_vocab);
            cdf[0] = probs[sorted_tokens[0]];
            for (int32_t j = 1; j < n_vocab; j++) {
                cdf[j] = cdf[j-1] + probs[sorted_tokens[j]];
            }

            int32_t rank = 0;
            for (; rank < n_vocab; rank++) {
                if (cdf[rank] > position) break;
            }
            if (rank >= n_vocab) rank = n_vocab - 1;

            int32_t token = sorted_tokens[rank];
            decompressed_tokens.push_back(token);

            // Calculate probability range
            double prob_before = (rank > 0) ? cdf[rank - 1] : 0.0;
            double token_prob = probs[token];

            // Update interval
            lo = lo + static_cast<uint64_t>(prob_before * width);
            hi = lo + std::max(1ULL, static_cast<uint64_t>(token_prob * width));

            // Add this token to KV cache for next iteration
            llama_decode(ctx, llama_batch_get_one(&token, 1));
            kv_cache_size++;

            // Renormalize
            while (true) {
                if (hi <= HALF) {
                    lo = lo << 1;
                    hi = hi << 1;
                    value = static_cast<uint32_t>(value) << 1;
                    if (bit_index < bits.size()) {
                        value = value | bits[bit_index];
                        bit_index++;
                    }
                } else if (lo >= HALF) {
                    lo = (lo - HALF) << 1;
                    hi = (hi - HALF) << 1;
                    value = (static_cast<uint32_t>(value) - static_cast<uint32_t>(HALF)) << 1;
                    if (bit_index < bits.size()) {
                        value = value | bits[bit_index];
                        bit_index++;
                    }
                } else if (lo >= QUARTER && hi <= THREE_QUARTERS) {
                    lo = (lo - QUARTER) << 1;
                    hi = (hi - QUARTER) << 1;
                    value = (static_cast<uint32_t>(value) - static_cast<uint32_t>(QUARTER)) << 1;
                    if (bit_index < bits.size()) {
                        value = value | bits[bit_index];
                        bit_index++;
                    }
                } else {
                    break;
                }
            }
        }

        // Detokenize
        std::string result;
        for (int32_t token : decompressed_tokens) {
            char buf[256];
            int32_t len = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
            if (len > 0) {
                result.append(buf, len);
            }
        }

        return result;
    }

    // Get model path
    const std::string& get_model_path() const {
        return model_path;
    }
};

} // namespace smol

PYBIND11_MODULE(_smol, m) {
    m.doc() = "Smol compression library";

    py::class_<smol::Compressor>(m, "Compressor")
        .def(py::init<const std::string&>(),
             py::arg("model_path"),
             "Create a new Compressor with the specified model")
        .def("compress", &smol::Compressor::compress,
             py::arg("data"),
             "Compress a string and return a bitstream")
        .def("decompress", &smol::Compressor::decompress,
             py::arg("bitstream"),
             "Decompress a bitstream and return a string")
        .def("__enter__", [](smol::Compressor &self) -> smol::Compressor& {
            return self;
        })
        .def("__exit__", [](smol::Compressor &self, py::object exc_type, py::object exc_value, py::object traceback) {
            // Destructor will be called automatically
            return false;
        })
        .def_property_readonly("model_path", &smol::Compressor::get_model_path,
             "Get the path to the loaded model");
}

#endif // SMOL_HPP
