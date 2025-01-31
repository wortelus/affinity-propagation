#include <chrono>
#include <iostream>

#include "affinity_propagation.h"

int main()
{
    auto start_time = std::chrono::high_resolution_clock::now();
    AffinityPropagation ap("mnist_test.csv", true);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout <<
        "T(parallel file read): " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() <<
        " ms" <<
        std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    ap.setSimilarityMatrix();
    end_time = std::chrono::high_resolution_clock::now();
    std::cout <<
        "T(parallel similarity matrix): " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() <<
        " ms" <<
        std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    int iters = ap.run(1);
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Iterations: " << iters << std::endl;
    std::cout <<
        "T(parallel affinity propagation): " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() <<
        " ms" <<
        std::endl;

    ap.C_matrix();

    ap.printClusterCounts();

    return 0;
}
