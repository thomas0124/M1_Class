#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_N 500001

void sieve(int* is_prime) {
    for (int i = 2; i < MAX_N; i++) is_prime[i] = 1;
    is_prime[0] = is_prime[1] = 0;
    for (int i = 2; i * i < MAX_N; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j < MAX_N; j += i) {
                is_prime[j] = 0;
            }
        }
    }
}

unsigned long long gcd(unsigned long long a, unsigned long long b) {
    while (b) {
        unsigned long long tmp = a % b;
        a = b;
        b = tmp;
    }
    return a;
}

unsigned long long mod_pow(unsigned long long a, unsigned long long b, unsigned long long m) {
    unsigned long long result = 1;
    a = a % m;
    while (b > 0) {
        if (b & 1) result = (result * a) % m;
        a = (a * a) % m;
        b >>= 1;
    }
    return result;
}

int fermat_test(unsigned long long n, int k) {
    if (n <= 2 || n % 2 == 0) return 0;
    for (int i = 0; i < k; i++) {
        unsigned long long a;
        do {
            a = 2 + rand() % (n - 2);
        } while (gcd(a, n) != 1);

        if (mod_pow(a, n - 1, n) != 1) return 0;
    }
    return 1;
}

int miller_rabin_test(unsigned long long n, int k) {
    if (n < 2 || n % 2 == 0) return 0;
    unsigned long long d = n - 1;
    int s = 0;

    while (d % 2 == 0) {
        d /= 2;
        s++;
    }

    for (int i = 0; i < k; i++) {
        unsigned long long a = 1 + rand() % (n - 2);
        unsigned long long x = mod_pow(a, d, n);
        if (x == 1 || x == n - 1) continue;

        int continue_outer = 0;
        for (int r = 1; r < s; r++) {
            x = mod_pow(x, 2, n);
            if (x == n - 1) {
                continue_outer = 1;
                break;
            }
        }
        if (continue_outer) continue;
        return 0;
    }
    return 1;
}

int is_carmichael(int n, int* is_prime) {
    if (is_prime[n] || n < 3 || n % 2 == 0) return 0;

    for (int a = 2; a < n; a++) {
        if (gcd(a, n) == 1 && mod_pow(a, n - 1, n) != 1) {
            return 0;
        }
    }
    return 1;
}

int main() {
    srand((unsigned int)time(NULL));
    clock_t start, end;
    start = clock();
    int* is_prime = (int*)malloc(sizeof(int) * MAX_N);
    int k = 10;
    sieve(is_prime);

    printf("Carmichael numbers less than or equal to %d:\n", MAX_N - 1);

    int count = 0;
    for (int i = 1; i < MAX_N; i++) {
        if (is_carmichael(i, is_prime)) {
            count++;
            printf("%d: ", i);
            printf("Fermat: %s, ", fermat_test(i, k) ? "Pass" : "Fail");
            printf("MR: %s\n", miller_rabin_test(i, k) ? "Pass" : "Fail");
        }
    }

    printf("Total Carmichael numbers found: %d\n", count);
    free(is_prime);

    end = clock();

    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %.2f seconds\n", elapsed_time);

    return 0;
}
