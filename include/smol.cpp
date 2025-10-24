#ifndef SMOL_HPP
#define SMOL_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <stdexcept>
#include "llama.h"

namespace py = pybind11;

namespace smol {

class Compressor {
private:
    llama_model* model;
    std::string model_path;

public:
    // Constructor - loads the model
    Compressor(const std::string& path) : model_path(path) {
        llama_model_params params = llama_model_default_params();
        model = llama_load_model_from_file(path.c_str(), params);

        if (!model) {
            throw std::runtime_error("Failed to load model: " + path);
        }
    }

    // Destructor - frees the model
    ~Compressor() {
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
        if (!model) {
            throw std::runtime_error("Model not loaded");
        }

        // TODO: Implement actual compression logic using the model
        // This is a stub that returns empty bit stream
        std::vector<uint8_t> bitstream;
        return bitstream;
    }

    // Decompress function - takes bit stream, returns string or throws
    std::string decompress(const std::vector<uint8_t>& bitstream) {
        if (!model) {
            throw std::runtime_error("Model not loaded");
        }

        // TODO: Implement actual decompression logic
        // This is a stub that throws if bitstream is empty
        if (bitstream.empty()) {
            throw std::runtime_error("Empty bitstream provided");
        }
        return "";
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
