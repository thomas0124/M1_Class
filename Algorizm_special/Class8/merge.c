#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define V_SIZE (1LL << 20)

long long *v;

void randomize(long long seed) {
    srand(seed);
    for (long long i = 0; i < V_SIZE; ++i) {
        v[i] = rand();
    }
}

void print_array(long long *arr, long long size) {
    for (long long i = 0; i < size; ++i) {
        printf("%lld ", arr[i]);
    }
    printf("\n");
}

void merge(long long *arr, long long left, long long mid, long long right) {
    long long n1 = mid - left + 1;
    long long n2 = right - mid;

    long long *L = malloc(sizeof(long long) * n1);
    long long *R = malloc(sizeof(long long) * n2);

    for (long long i = 0; i < n1; i++) L[i] = arr[left + i];
    for (long long j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    long long i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

void merge_sort(long long *arr, long long left, long long right) {
    if (left < right) {
        long long mid = left + (right - left) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

int main() {
    v = malloc(sizeof(long long) * V_SIZE);
    if (!v) {
        fprintf(stderr, "メモリ確保失敗\n");
        return 1;
    }

    randomize(1);

    print_array(v, 16);

    clock_t start = clock();
    merge_sort(v, 0, V_SIZE - 1);
    clock_t end = clock();

    print_array(v, 16);

    printf("MergeSort 実行時間: %.6f 秒\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(v);
    return 0;
}