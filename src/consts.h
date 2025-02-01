//
// Created by wortelus on 31.01.2025.
//

#ifndef CONSTS_H
#define CONSTS_H

// #define USE_DAMPING 1

// If you want individual times for R(i, k), A(k, k) and A(i, k), uncomment following line
// Pokud chceme jednotlivé časy pro R(i, k), A(k, k) a A(i, k), pak odkomentuj následující řádek
#define FINE_TIMES 1

// MNIST dataset has 28x28 pixels
static constexpr size_t MNIST_PIXELS = 28;

// Number of features in dataset (excluding label)
static constexpr size_t FEATURE_COUNT = MNIST_PIXELS * MNIST_PIXELS;

// CSV line for MNIST dataset is expected to be aprox. 1850 bytes long
// So we reserve 1500 bytes per thread to avoid reallocations (lower size -> larger allocation)
static constexpr size_t THREAD_RESERVE_COEFFICIENT = 1500;

#endif //CONSTS_H
