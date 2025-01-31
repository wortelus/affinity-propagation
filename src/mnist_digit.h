//
// Created by wortelus on 31.01.2025.
//

#ifndef MNIST_DIGIT_H
#define MNIST_DIGIT_H
#include <cstdint>

#include "consts.h"

struct MnistDigit
{
    uint8_t label;
    uint8_t pixels[FEATURE_COUNT];
};

#endif //MNIST_DIGIT_H
