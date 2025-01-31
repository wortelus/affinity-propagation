//
// Created by wortelus on 31.01.2025.
//

#ifndef AFFINITY_PROPAGATION_H
#define AFFINITY_PROPAGATION_H

#include <vector>

#include "mnist_digit.h"


class AffinityPropagation {
    double lambda = 0.5; 
    
    int digit_count;
    int matrix_size;
    int half_matrix_size;
    int half_matrix_dia_size;

    std::vector<MnistDigit> digits;

    std::vector<double> similarity_matrix;
    std::vector<double> responsibility_matrix;
    std::vector<double> availability_matrix;

    std::vector<double> c_matrix;


    static size_t getFileSize(const char* filename);
    inline double* matrixAt(std::vector<double>& matrix, int i, int j) const;
    inline static double* halfMatrixAt(std::vector<double>& matrix, int i, int j);
    inline static double* halfMatrixDiagAt(std::vector<double>& matrix, int i, int j);
    static double* halfMatrixDiagAtChecked(std::vector<double>& matrix, int i, int j);

public:
    AffinityPropagation(const char* filename, bool parallel);
    void setSimilarityMatrix();
    int run(int max_iter);
    void C_matrix();
    void printClusterCounts();
};


#endif //AFFINITY_PROPAGATION_H
