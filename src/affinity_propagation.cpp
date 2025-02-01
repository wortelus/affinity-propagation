//
// Created by wortelus on 31.01.2025.
//


#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <limits>

#include <sys/stat.h>
#include <omp.h>
#include <unordered_map>

#include "affinity_propagation.h"
#include "consts.h"

// AVX2
#include <immintrin.h>

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

inline double* AffinityPropagation::matrixAt(std::vector<double>& matrix, int i, int j) const
{
    return &matrix[i * digit_count + j];
}

/// IMPORTANT: Ensure matrix doesn't reallocate, doesn't run out of bounds, and j > i
inline double* AffinityPropagation::halfMatrixAt(std::vector<double>& matrix, const int i, const int j)
{
    return &matrix[j * (j - 1) / 2 + i];
}


// /// IMPORTANT: Ensure matrix doesn't reallocate, doesn't run out of bounds, and j > i
// inline double* AffinityPropagation::halfMatrixDiagAt(std::vector<double>& matrix, const int i, const int j)
// {
//     return &matrix[j * (j + 1) / 2 + i];
// }
//
// inline double* AffinityPropagation::halfMatrixDiagAtChecked(std::vector<double>& matrix, const int i, const int j)
// {
//     if (i <= j)
//         return &matrix[j * (j + 1) / 2 + i];
//     else
//         return &matrix[i * (i + 1) / 2 + j];
// }

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
    this->similarity_matrix = std::vector(matrix_size, 0.0);
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
            *matrixAt(similarity_matrix, i, j) = -distance;
            *matrixAt(similarity_matrix, j, i) = -distance;
        }
    }

    if (!median_vec.empty())
    {
        std::ranges::nth_element(
            median_vec,
            median_vec.begin() + median_vec.size() / 2
        );

        double median_similarity = median_vec[median_vec.size() / 2];

        // USING MINIMUM INSTEAD OF MEDIAN
        median_similarity = median_vec[median_vec.size() - 1];

        // Set median to the diagonal
#pragma omp parallel for default(none) shared(similarity_matrix, digit_count, median_similarity)
        for (int i = 0; i < this->digit_count; ++i)
        {
            // this->similarity_matrix[i * this->digit_count + i] = -median_similarity;
            *matrixAt(similarity_matrix, i, i) = -median_similarity;
        }
    }
}

int AffinityPropagation::run(const int max_iter)
{
    this->responsibility_matrix = std::vector(matrix_size, 0.0);
    this->availability_matrix = std::vector(matrix_size, 0.0);

    std::vector<double> R_old = this->responsibility_matrix;
    std::vector<double> A_old = this->availability_matrix;

    int iter = 0;
    for (; iter < max_iter; ++iter)
    {
        // 1)
        // R(i, k)
        //
#ifdef FINE_TIMES
        auto start_time = std::chrono::high_resolution_clock::now();
#endif
#pragma omp parallel for default(none) shared(similarity_matrix, R_old, A_old, responsibility_matrix, digit_count)
        for (int i = 0; i < digit_count; ++i)
        {
            //
            // Responsibility matrix update
            //

            // precompute
            // max_kk(a(i, kk) + s(i, kk))
            std::vector<double> max_i_kk(digit_count);

            // Either OpenMP SIMD or AVX2
            // (pouze jako malá demonstrace :)) rozdíl v rychlosti jsem nezaznamenal
#ifdef __AVX2__
            int kk = 0;
            // process 4 doubles at a time
            for (; kk + 3 < digit_count; kk += 4)
            {
                __m256d a_vals = _mm256_loadu_pd(&A_old[i * digit_count + kk]);
                __m256d s_vals = _mm256_loadu_pd(&similarity_matrix[i * digit_count + kk]);
                __m256d sum_vals = _mm256_add_pd(a_vals, s_vals);
                _mm256_storeu_pd(&max_i_kk[kk], sum_vals);
            }

            // Remainder for anything not a multiple of 4
            for (; kk < digit_count; ++kk)
            {
                max_i_kk[kk] = A_old[i * digit_count + kk] + similarity_matrix[i * digit_count + kk];
            }
#else
#pragma omp simd
            for (int kk = 0; kk < digit_count; ++kk)
            {
                const auto a_i_kk = *matrixAt(A_old, i, kk);
                const auto s_i_kk = *matrixAt(similarity_matrix, i, kk);
                max_i_kk[kk] = a_i_kk + s_i_kk;
            }
#endif


            // Choose first & second max from max_i_kk
            int first_index = -1;
            double first = -std::numeric_limits<double>::infinity();
            double second = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < digit_count; ++k)
            {
                if (max_i_kk[k] > first)
                {
                    second = first;
                    first = max_i_kk[k];
                    first_index = k;
                }
                else if (max_i_kk[k] > second)
                {
                    second = max_i_kk[k];
                }
            }

            for (int k = 0; k < digit_count; ++k)
            {
                const double chosen_max_i_kk = k != first_index ? first : second;

                const auto s_i_k = *matrixAt(similarity_matrix, i, k);
                const double raw_r_ik = s_i_k - chosen_max_i_kk;

                // Add damping
#ifdef USE_DAMPING
                const double old_r_ik = *matrixAt(R_old, i, k);
                const double new_r_ik = (1.0 - lambda) * raw_r_ik + lambda * old_r_ik;
                *matrixAt(responsibility_matrix, i, k) = new_r_ik;
#else
                *matrixAt(responsibility_matrix, i, k) = raw_r_ik;
#endif
            }
        }
#ifdef FINE_TIMES
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << "T(parallel R(i, k) matrix): " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() <<
            " ms" <<
            std::endl;
#endif

        // 1b)
        // Update R(i, k)
        //
        R_old = this->responsibility_matrix;

        // 2)
        // A(k, k)
        //
#ifdef FINE_TIMES
        start_time = std::chrono::high_resolution_clock::now();
#endif
#pragma omp parallel for default(none) shared(similarity_matrix, R_old, A_old, availability_matrix, digit_count)
        for (int k = 0; k < digit_count; ++k)
        {
            // sum(max(0, R(ii, k))) for ii != k
            double sum = 0.0;
            for (int ii = 0; ii < digit_count; ++ii)
            {
                if (ii == k)
                    continue;

                const auto r_ii_k = *matrixAt(R_old, ii, k);

                sum += std::max(
                    0.0,
                    r_ii_k
                );
            }

            const double raw_a_kk = sum;

            // Add damping
#ifdef USE_DAMPING
            const double old_a_kk = *matrixAt(A_old, k, k);
            const double new_a_kk = (1.0 - lambda) * raw_a_kk + lambda * old_a_kk;
            *matrixAt(availability_matrix, k, k) = new_a_kk;
#else
            *matrixAt(availability_matrix, k, k) = raw_a_kk;
#endif
        }
#ifdef FINE_TIMES
        end_time = std::chrono::high_resolution_clock::now();
        std::cout << "T(parallel A(k, k) matrix): " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() <<
            " ms" <<
            std::endl;
#endif


        // 3)
        // A(i, k) for i != k
        //
#ifdef FINE_TIMES
        start_time = std::chrono::high_resolution_clock::now();
#endif
#pragma omp parallel for default(none) shared(similarity_matrix, R_old, A_old, responsibility_matrix, availability_matrix, digit_count)
        for (int k = 0; k < digit_count; ++k)
        {
            // Sum for A(i, k)

            // R(k, k) + sum(max(0, R(ii, k))) for ii != i
            //
            // R(k, k)
            const auto r_k_k = *matrixAt(R_old, k, k);
            // sum_max_R_ii_k
            double sum = r_k_k;
            for (int ii = 0; ii < digit_count; ++ii)
            {
                // two conditions:
                // ii != k
                // ii != i --- WARNING, MUST SUBTRACK R(i, k) FROM SUM LATER
                if (ii == k)
                    continue;

                const auto r_ii_k = *matrixAt(R_old, ii, k);

                sum += std::max(
                    0.0,
                    r_ii_k
                );
            }

            for (int i = 0; i < digit_count; ++i)
            {
                // Prevent overwrite of diagonal (A(k, k) is calculated next)
                if (i == k)
                    continue;

                // SUBTRACT R(i, k) FROM SUM (ii != i)
                sum -= std::max(0., *matrixAt(R_old, i, k));
                // min(0, R(k, k) + sum(max(0, R(ii, k))) for ii != i)
                const double raw_A = std::min(
                    0.0,
                    sum
                );

                // Add damping
#ifdef USE_DAMPING
                const double old_A = *matrixAt(A_old, i, k);
                const double new_A = (1.0 - lambda) * raw_A + lambda * old_A;
                *matrixAt(availability_matrix, i, k) = new_A;
#else
                *matrixAt(availability_matrix, i, k) = raw_A;
#endif
            }
        }
#ifdef FINE_TIMES
        end_time = std::chrono::high_resolution_clock::now();
        std::cout << "T(parallel A(i, k) matrix): " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() <<
            " ms" <<
            std::endl;
#endif

        // 3b)
        // Update A(i, k) and A(k, k)
        //
        A_old = this->availability_matrix;
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

void AffinityPropagation::printClusterCounts()
{
    std::unordered_map<int, int> counts;
    for (int i = 0; i < digit_count; ++i)
    {
        int highest_c = -1;
        double max_c = -std::numeric_limits<double>::infinity();

        for (int k = 0; k < digit_count; ++k)
        {
            if (*matrixAt(c_matrix, i, k) > max_c)
            {
                max_c = *matrixAt(c_matrix, i, k);
                highest_c = k;
            }
        }
        counts[highest_c]++;
    }

    std::cout << "Cluster Counts:" << std::endl;
    for (const auto& pair : counts)
    {
        std::cout << "Cluster " << pair.first << ": " << pair.second << " elements" << std::endl;
    }

    std::cout << "Total clusters: " << counts.size() << std::endl;
}
