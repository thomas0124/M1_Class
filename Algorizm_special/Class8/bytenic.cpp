#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <cmath>

#define SHIFTSIZE 20
#define V_SIZE (1LL << SHIFTSIZE)

std::vector<long long> v(V_SIZE);

void randomize(long long seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<long long> dist(0, std::numeric_limits<long long>::max());
    for (long long i = 0; i < V_SIZE; ++i) {
        v[i] = dist(rng);
    }
}

void swap(long long &a, long long &b) {
    long long tmp = a;
    a = b;
    b = tmp;
}

void swapping(long long i, long long j) {
    long long swap_distance = 1LL << (i - j);

    #pragma omp parallel for
    for (long long v_i = 0; v_i < V_SIZE / 2; ++v_i) {
        long long low = v_i & (swap_distance - 1);
        long long swap_pos = (v_i << 1) - low;
        int is_upper = (swap_pos & (2LL << i)) == 0;

        long long idx1 = swap_pos;
        long long idx2 = swap_pos | swap_distance;

        if ((v[idx1] > v[idx2]) == is_upper) {
            swap(v[idx1], v[idx2]);
        }
    }
}

void bitonic_sort() {
    long long log_vsize = std::log2(V_SIZE);
    for (long long i = 0; i < log_vsize; ++i) {
        for (long long j = 0; j <= i; ++j) {
            swapping(i, j);
        }
    }
}

void print_array(long long size) {
    for (long long i = 0; i < size; ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    try {
        randomize(1);
    } catch (const std::exception &e) {
        std::cerr << "乱数生成に失敗しました: " << e.what() << std::endl;
        return 1;
    }

    print_array(16);

    double start_time = omp_get_wtime();
    bitonic_sort();
    double end_time = omp_get_wtime();

    print_array(16);

    std::cout << "BitonicSort 実行時間: " << (end_time - start_time) << " 秒" << std::endl;

    return 0;
}