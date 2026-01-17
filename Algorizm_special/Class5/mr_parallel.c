#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>
#include <omp.h>

int MR_test(mpz_t n, int hanpuku, gmp_randstate_t rand_state) {
    if (mpz_cmp_ui(n, 2) == 0) return 1;
    if (mpz_cmp_ui(n, 1) == 0 || mpz_even_p(n)) return 0;

    mpz_t d, a, y, z, n_minus1, r;
    mpz_inits(d, a, y, z, n_minus1, r, NULL);
    mpz_sub_ui(n_minus1, n, 1);
    mpz_set(d, n_minus1);

    int s = 0;
    while (mpz_even_p(d)) {
        mpz_divexact_ui(d, d, 2);
        s++;
    }

    for (int i = 0; i < hanpuku; i++) {
        mpz_sub_ui(r, n, 3);
        mpz_urandomm(a, rand_state, r);
        mpz_add_ui(a, a, 2);

        mpz_powm(y, a, d, n);
        mpz_set(z, y);

        int j;
        for (j = 0; j < s; j++) {
            if (mpz_cmp_ui(z, 1) == 0 || mpz_cmp(z, n_minus1) == 0) break;
            mpz_powm_ui(z, z, 2, n);
        }

        if (mpz_cmp_ui(y, 1) != 0 && mpz_cmp(z, n_minus1) != 0) {
            mpz_clears(d, a, y, z, n_minus1, r, NULL);
            return 0;
        }
    }

    mpz_clears(d, a, y, z, n_minus1, r, NULL);
    return 1;
}

int main() {
    clock_t start, end;
    start = clock();

    const unsigned long long num = 500000000;
    int hanpuku = 10;

    int prime_count = 0;
    #pragma omp parallel
    {
        gmp_randstate_t rand_state;
        gmp_randinit_mt(rand_state);
        gmp_randseed_ui(rand_state, time(NULL) + omp_get_thread_num());

        mpz_t val;
        mpz_init(val);

        mpz_t start_val;
        mpz_init(start_val);
        mpz_ui_pow_ui(start_val, 10, 70);

        #pragma omp for reduction(+:prime_count) schedule(dynamic)
        for (unsigned long long i = 0; i < num; i++) {
            mpz_add_ui(val, start_val, i);
            if (MR_test(val, hanpuku, rand_state)) {
                prime_count++;
            }
            if ((i+1) % 1000000 == 0) {
                #pragma omp critical
                {
                    gmp_printf("Thread %d checked %llu numbers. Current: %Zd\n", omp_get_thread_num(), i+1, val);
                }
            }
        }

        mpz_clear(val);
        mpz_clear(start_val);
        gmp_randclear(rand_state);
    }

    printf("Total probable primes: %d\n", prime_count);

    end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %.2f seconds\n", elapsed_time);

    return 0;
}
