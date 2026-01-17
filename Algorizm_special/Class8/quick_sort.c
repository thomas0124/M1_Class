#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cstdlib> 
#include <limits>  

#define V_SIZE (1LL << 20)

std::vector<long long> v(V_SIZE);

void randomize(long long seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<long long> dist(0, std::numeric_limits<long long>::max());
    for (long long i = 0; i < V_SIZE; ++i) {
        v[i] = dist(rng);
    }
}

void print_array(const std::vector<long long> &arr, long long size) {
    for (long long i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void swap(long long &a, long long &b) {
    long long tmp = a;
    a = b;
    b = tmp;
}

long long partition(std::vector<long long> &arr, long long left, long long right) {
    long long pivot = arr[right];
    long long i = left - 1;
    for (long long j = left; j < right; ++j) {
        if (arr[j] <= pivot) {
            ++i;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[right]);
    return i + 1;
}

void quick_sort(std::vector<long long> &arr, long long left, long long right) {
    if (left < right) {
        long long pi = partition(arr, left, right);
        quick_sort(arr, left, pi - 1);
        quick_sort(arr, pi + 1, right);
    }
}

int main() {
    try {
        randomize(1);
    } catch (const std::exception &e) {
        std::cerr << "乱数生成に失敗しました: " << e.what() << std::endl;
        return 1;
    }

    print_array(v, 16);

    std::clock_t start = std::clock();
    quick_sort(v, 0, V_SIZE - 1);
    std::clock_t end = std::clock();

    print_array(v, 16);

    std::cout << "QuickSort 実行時間: " << (double)(end - start) / CLOCKS_PER_SEC << " 秒" << std::endl;

    return 0;
}
