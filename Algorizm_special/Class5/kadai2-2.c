#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_N 500001

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

int fermat_test(unsigned long long n) {
    if (n <= 2 || n % 2 == 0) return 0;
    return mod_pow(2, n - 1, n) == 1;
}

int miller_rabin_test(unsigned long long n) {
    if (n < 2 || n % 2 == 0) return 0;
    unsigned long long d = n - 1;
    int s = 0;

    while (d % 2 == 0) {
        d /= 2;
        s++;
    }

    unsigned long long a = 2;
    unsigned long long x = mod_pow(a, d, n);
    if (x == 1 || x == n - 1) return 1;

    for (int r = 1; r < s; r++) {
        x = mod_pow(x, 2, n);
        if (x == n - 1) return 1;
    }

    return 0;
}

unsigned long long gcd(unsigned long long a, unsigned long long b) {
    while (b) {
        unsigned long long tmp = a % b;
        a = b;
        b = tmp;
    }
    return a;
}

int is_carmichael(int n) {
    for (int a = 2; a < n; a++) {
        if (gcd(a, n) == 1 && mod_pow(a, n - 1, n) != 1) {
            return 0;
        }
    }
    return 1;
}

int main() {
    printf("Carmichael number candidates up to %d:\n", MAX_N - 1);

    int count = 0;
    for (int i = 1; i < MAX_N; i++) {
        if (is_carmichael(i)) {
            count++;
            printf("%d: ", i);
            printf("Fermat: %s, ", fermat_test(i) ? "Pass" : "Fail");
            printf("MR: %s\n", miller_rabin_test(i) ? "Pass" : "Fail");
        }
    }

    printf("Total Carmichael numbers found: %d\n", count);
    return 0;
}
