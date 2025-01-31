//
// Created by wortelus on 31.01.2025.
//

#include "affinity_propagation.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <limits>

#include <sys/stat.h>
#include <omp.h>
#include <unordered_map>

#include "consts.h"

///
/// @brief Get file size
/// @param filename File path
/// @return File size in bytes
///
size_t AffinityPropagation::getFileSize(const char* filename)
{
    struct stat st{};
    if (stat(filename, &st) != 0)
    {
        perror("stat() failed");
        exit(EXIT_FAILURE);
    }
    return st.st_size;
}

double* AffinityPropagation::matrixAt(std::vector<double>& matrix, int i, int j) const
{
    return &matrix[i * digit_count + j];
}

/// IMPORTANT: Ensure matrix doesn't reallocate, doesn't run out of bounds, and j > i
double* AffinityPropagation::halfMatrixAt(std::vector<double>& matrix, const int i, const int j)
{
    return &matrix[j * (j - 1) / 2 + i];
}


/// IMPORTANT: Ensure matrix doesn't reallocate, doesn't run out of bounds, and j > i
inline double* AffinityPropagation::halfMatrixDiagAt(std::vector<double>& matrix, const int i, const int j)
{
    return &matrix[j * (j + 1) / 2 + i];
}

inline double* AffinityPropagation::halfMatrixDiagAtChecked(std::vector<double>& matrix, const int i, const int j)
{
    if (i <= j)
        return &matrix[j * (j + 1) / 2 + i];
    else
        return &matrix[i * (i + 1) / 2 + j];
}

AffinityPropagation::AffinityPropagation(const char* filename, bool parallel)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Get file size for parallel processing
    size_t file_size = getFileSize(filename);
    if (file_size <= 0)
    {
        std::cerr << "Error reading file while getting size." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Skip first line (CSV header)
    std::string dummy;
    std::getline(file, dummy);
    size_t start_offset = file.tellg();

    int num_threads = parallel ? omp_get_max_threads() : 1;
    std::vector<std::vector<MnistDigit>> thread_digits(num_threads);
    for (auto& v : thread_digits)
    {
        v.reserve(file_size / static_cast<size_t>(num_threads) / THREAD_RESERVE_COEFFICIENT);
    }

    // Chunk size
    // + (num_threads - 1) to account for rounding up
    size_t chunk_size = (file_size - start_offset + num_threads - 1) / num_threads;

    // std::cout << "Filesize: " << file_size << std::endl;
    //
    // Parallel processing
    //
#pragma omp parallel if(parallel) default(none) shared(filename, file_size, start_offset, chunk_size, num_threads, thread_digits, std::cerr, std::cout)
    {
        int tid = omp_get_thread_num();
        size_t start = start_offset + tid * chunk_size;
        size_t end = tid != num_threads - 1 ? std::min(start + chunk_size, file_size) : file_size;

        // DEBUG
        // #pragma omp critical
        //         {
        //             std::cout << "Thread " << tid << " processing chunk " << start << " - " << end << std::endl;
        //         }

        // Each thread opens its own file stream
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs)
        {
#pragma omp critical
            {
                std::cerr << "Thread " << tid << " cannot open file." << std::endl;
            }
            // #pragma omp cancel parallel
        }

        // Seek to the beginning of the chunk
        ifs.seekg(static_cast<std::streamoff>(start));

        // Check if we are not at the very beginning of the file
        if (start != 0)
        {
            // Peek the previous character
            ifs.seekg(-1, std::ios::cur);
            char prev;
            ifs.get(prev);

            // If we are at the beginning of the line, our read position is at the correct position
            // If we are at the at middle/end of the line, we skip the rest of the line (part of previous chunk)
            if (prev == '\n')
            {
                // DEBUG
                // #pragma omp critical
                //                 {
                //                     std::cerr << "Previous char was '\\n', including line" << std::endl;
                //                 }
            }
            else
            {
                // Skip the rest of the line
                std::string dummy;
                std::getline(ifs, dummy);
            }
        }

        // Read lines and process MNIST 28x28
        std::streampos current_pos = ifs.tellg();
        std::string line;
        while (current_pos >= 0 && current_pos < end && std::getline(ifs, line))
        {
            current_pos = ifs.tellg();

            // Convert std::string to mutable C-string
            char* c_str = &line[0];
            // Read label first
            char* token = std::strtok(c_str, ",");

            MnistDigit digit{};
            digit.label = static_cast<uint8_t>(std::stoi(token));

            // Read 28x28 pixels
            for (unsigned char& pixel : digit.pixels)
            {
                token = std::strtok(nullptr, ",");
                if (token == nullptr)
                {
                    std::cerr << "Thread " << tid << " failed to parse line: " << line << std::endl;
                    break;
                }
                pixel = static_cast<uint8_t>(std::stoi(token));
            }

            thread_digits[tid].push_back(digit);
        }


        // Close file stream in this thread
        ifs.close();
    }
    //
    // End of parallel section
    //

    // Unify results into one vector
    std::vector<MnistDigit> digits;
    {
        size_t total = 0;
        for (auto& v : thread_digits)
        {
            total += v.size();
        }
        digits.reserve(total);
        for (auto& v : thread_digits)
        {
            digits.insert(digits.end(), v.begin(), v.end());
        }
    }

    this->digit_count = static_cast<int>(digits.size());
    this->digits = std::move(digits);
    this->matrix_size = this->digit_count * this->digit_count;
    this->half_matrix_size = this->digit_count * (this->digit_count - 1) / 2;
    this->half_matrix_dia_size = digit_count + half_matrix_size;
}

void AffinityPropagation::setSimilarityMatrix()
{
    this->similarity_matrix = std::vector(half_matrix_dia_size, 0.0);
    std::vector median_vec(half_matrix_size, 0.0);

    // #pragma omp parallel for default(none) shared(digits, similarity_matrix, median_vec)
    for (int i = 0; i < this->digit_count; ++i)
    {
        // Only one side of the diagonal
        for (int j = i + 1; j < this->digit_count; ++j)
        {
            double distance = 0.0;

            // Pixel-wise difference
            for (int k = 0; k < FEATURE_COUNT; ++k)
            {
                const double diff = static_cast<double>(digits[i].pixels[k]) - static_cast<double>(digits[j].pixels[k]);
                distance += diff * diff;
            }

            *halfMatrixAt(median_vec, i, j) = distance;
            *halfMatrixDiagAt(similarity_matrix, i, j) = -distance;
        }
    }

    if (!median_vec.empty())
    {
        std::ranges::nth_element(
            median_vec,
            median_vec.begin() + median_vec.size() / 2
        );

        double median_similarity = median_vec[median_vec.size() / 2];
        median_similarity = 22; // DEBUG

        // Set median to the diagonal
#pragma omp parallel for default(none) shared(similarity_matrix, digit_count, median_similarity)
        for (int i = 0; i < this->digit_count; ++i)
        {
            // this->similarity_matrix[i * this->digit_count + i] = -median_similarity;
            *halfMatrixDiagAt(similarity_matrix, i, i) = -median_similarity;
        }
    }
}

int AffinityPropagation::run(const int max_iter)
{
    this->responsibility_matrix = std::vector(matrix_size, 0.0);
    this->availability_matrix = std::vector(matrix_size, 0.0);

    std::vector<double> R_matrix = this->responsibility_matrix;
    std::vector<double> A_matrix = this->availability_matrix;

    int iter = 0;
    for (; iter < max_iter; ++iter)
    {
#pragma omp parallel for default(none) shared(similarity_matrix, R_matrix, A_matrix, responsibility_matrix, availability_matrix, digit_count)
        for (int i = 0; i < digit_count; ++i)
        {
            for (int k = 0; k < digit_count; ++k)
            {
                //
                // Responsibility matrix update
                //

                // max_kk(a(i, kk) + s(i, kk))
                double max_kk = -std::numeric_limits<double>::infinity();
                for (int kk = 0; kk < digit_count; ++kk)
                {
                    // kk != k
                    if (kk == k)
                        continue;

                    const auto a_i_kk = *matrixAt(A_matrix, i, kk);
                    const auto s_i_kk = *halfMatrixDiagAtChecked(similarity_matrix, i, kk);
                    max_kk = std::max(
                        max_kk,
                        a_i_kk + s_i_kk
                    );
                }

                // R(i, k) = s(i, k) - max_kk(a(i, kk) + s(i, kk))
                const auto s_i_k = *halfMatrixDiagAtChecked(similarity_matrix, i, k);
                *matrixAt(responsibility_matrix, i, k) = s_i_k - max_kk;
            }
        }


        //
        // A(k, k)
        //
#pragma omp parallel for default(none) shared(similarity_matrix, R_matrix, availability_matrix, digit_count)
        for (int k = 0; k < digit_count; ++k)
        {
            // sum(max(0, R(ii, k))) for ii != k
            double sum = 0.0;
            for (int ii = 0; ii < digit_count; ++ii)
            {
                if (ii == k)
                    continue;

                const auto r_ii_k = *matrixAt(R_matrix, ii, k);

                sum += std::max(
                    0.0,
                    r_ii_k
                );
            }
            
            *matrixAt(availability_matrix, k, k) = sum;
        }

#pragma omp parallel for default(none) shared(similarity_matrix, R_matrix, A_matrix, responsibility_matrix, availability_matrix, digit_count)
        for (int i = 0; i < digit_count; ++i)
        {
            for (int k = 0; k < digit_count; ++k)
            {
                //
                // Availability matrix update
                //

                // Prevent overwrite of diagonal (A(k, k) is calculated next)
                if (i == k)
                    continue;

                // R(k, k) + sum(max(0, R(ii, k))) for ii != i
                //
                // R(k, k)
                const auto r_k_k = *matrixAt(R_matrix, k, k);
                // sum_max_R_ii_k
                double sum = r_k_k;
                for (int ii = 0; ii < digit_count; ++ii)
                {
                    // ii != k also added by me
                    if (ii == i || ii == k)
                        continue;

                    const auto r_ii_k = *matrixAt(R_matrix, ii, k);

                    sum += std::max(
                        0.0,
                        r_ii_k
                    );
                }

                // min(0, R(k, k) + sum(max(0, R(ii, k))) for ii != i)
                *matrixAt(availability_matrix, i, k) = std::min(
                    0.0,
                    sum
                );
            }
        }

        R_matrix = this->responsibility_matrix;
        A_matrix = this->availability_matrix;

// #pragma omp critical
//         {
//             std::cout << "Swapped from iter " << iter << " into " << iter + 1 << std::endl;
//         }
    }

    return iter;
}

void AffinityPropagation::C_matrix()
{
    this->c_matrix = std::vector<double>(digit_count * digit_count, 0.0);
#pragma omp parallel for default(none) shared(c_matrix, responsibility_matrix, availability_matrix, digit_count)
    for (int i = 0; i < digit_count; ++i)
    {
        for (int k = 0; k < digit_count; ++k)
        {
            *matrixAt(c_matrix, i, k) =
                *matrixAt(responsibility_matrix, i, k) +
                *matrixAt(availability_matrix, i, k);
        }
    }
}

void AffinityPropagation::printClusterCounts() {
    std::unordered_map<int, int> counts;
    for (int i = 0; i < digit_count; ++i) {
        int highest_c = -1;
        double max_c = -std::numeric_limits<double>::infinity();
        
        for (int k = 0; k < digit_count; ++k) {
            if (*matrixAt(c_matrix, i, k) > max_c) {
                max_c = *matrixAt(c_matrix, i, k);
                highest_c = k;
            }
        }
        counts[highest_c]++;
    }
    
    std::cout << "Cluster Counts:" << std::endl;
    for (const auto& pair : counts) {
        std::cout << "Cluster " << pair.first << ": " << pair.second << " elements" << std::endl;
    }

    std::cout << "Total clusters: " << counts.size() << std::endl;
}
