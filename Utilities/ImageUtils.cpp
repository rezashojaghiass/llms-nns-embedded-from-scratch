#include "ImageUtils.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>

Tensor4D<float> loadImageAsTensor(const std::string& filename) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR); // Loads as BGR, 8-bit
    if (img.empty()) throw std::runtime_error("Failed to load image: " + filename);

    size_t rows = img.rows;
    size_t cols = img.cols;
    size_t channels = img.channels();
    Tensor4D<float> tensor({ 1, channels, rows, cols });

    for (size_t c = 0; c < channels; ++c)
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                tensor(0, c, i, j) = static_cast<float>(img.at<cv::Vec3b>(i, j)[c]);
    return tensor;
}
