#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#include <time.h>

int MR_test(mpz_t n, int hanpuku, gmp_randstate_t rand_state) {
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
    double start, end;
    start = omp_get_wtime();

    mpz_t base_val;
    mpz_init(base_val);
    mpz_ui_pow_ui(base_val, 10, 70); // 10^70
    if (mpz_even_p(base_val)) {
        mpz_add_ui(base_val, base_val, 1);
    }

    const unsigned long long num = 500000000;
    int hanpuku = 10;
    int prime_count = 0;

    const int small_primes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29};
    const int small_prime_count = sizeof(small_primes) / sizeof(small_primes[0]);

    #pragma omp parallel
    {
        int local_count = 0;
        gmp_randstate_t rand_state;
        gmp_randinit_mt(rand_state);
        gmp_randseed_ui(rand_state, time(NULL) ^ omp_get_thread_num());

        mpz_t val;
        mpz_init(val);

        #pragma omp for
        for (unsigned long long i = 0; i < num; i+=2) {
            mpz_set(val, base_val);
            mpz_add_ui(val, val, i);

            int divisible = 0;
            for (int j = 0; j < small_prime_count; j++) {
                if (mpz_divisible_ui_p(val, small_primes[j])) {
                    divisible = 1;
                    break;
                }
            }

            if (!divisible && MR_test(val, hanpuku, rand_state)) {
                local_count++;
            }
        }

        #pragma omp atomic
        prime_count += local_count;

        mpz_clear(val);
        gmp_randclear(rand_state);
    }

    mpz_clear(base_val);

    printf("Total probable primes: %d\n", prime_count);

    end = omp_get_wtime();
    printf("Execution time: %.2f seconds\n", end - start);
    return 0;
}