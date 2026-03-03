#pragma once
#include <string>
#include "Tensor4D.h"

// Loads an image file into a Tensor4D<float> with shape {1, channels, rows, cols}
Tensor4D<float> loadImageAsTensor(const std::string& filename);
 
