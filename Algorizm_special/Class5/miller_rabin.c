#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>

int MR_test(mpz_t n, int hanpuku, gmp_randstate_t rand_state) {
    if (mpz_cmp_ui(n, 2) == 0) return 1; //nが2であるかの判定。 2は素数なので素数(1)を返す
    if (mpz_cmp_ui(n, 1) == 0 || mpz_even_p(n)) return 0; //nが1であるかの判定、または偶数の判定。どちらかを満たしたら素数ではないので素数ではないと返す(0)

    mpz_t d, a, y, z, n_minus1, r;
    mpz_inits(d, a, y, z, n_minus1, r, NULL);　//変数d, a, y, z, n_minus1, rに初期値NULLを代入
    mpz_sub_ui(n_minus1, n, 1); //n_minus1にnから1を引いた値を代入
    mpz_set(d, n_minus1); //dにn_minus1の値を代入

    int s = 0;
    /* dが偶数の場合、dを2で割り続け、sを1ずつ増やす */
    while (mpz_even_p(d)) { 
        mpz_divexact_ui(d, d, 2);
        s++;
    }
    /* hanpuku回試行 */
    for (int i = 0; i < hanpuku; i++) {
        mpz_sub_ui(r, n, 3); //rにnから3を引いた値を代入
        mpz_urandomm(a, rand_state, r); //aにrand_stateからrの範囲でランダムに選んだ値を代入
        mpz_add_ui(a, a, 2); //aにa+2を代入

        mpz_powm(y, a, d, n); //yにa^d mod nを代入
        mpz_set(z, y); //zにyの値を代入

        int j;
        for (j = 0; j < s; j++) {
            if (mpz_cmp_ui(z, 1) == 0 || mpz_cmp(z, n_minus1) == 0) break; //zが1かn_minus1と等しい場合、ループを抜ける
            mpz_powm_ui(z, z, 2, n); //zにz^2 mod nを代入
        }

        if (mpz_cmp_ui(y, 1) != 0 && mpz_cmp(z, n_minus1) != 0) {
            mpz_clears(d, a, y, z, n_minus1, r, NULL); //変数d, a, y, z, n_minus1, rに初期値NULLを代入
            return 0;
        }
    }

    mpz_clears(d, a, y, z, n_minus1, r, NULL);
    return 1;
}

int main() {
    clock_t start, end; //時間計測用の変数
    start = clock();
    mpz_t val;
    mpz_init(val); //変数valに初期値NULLを代入
    gmp_randstate_t rand_state;
    gmp_randinit_mt(rand_state); //rand_stateに初期値を代入
    gmp_randseed_ui(rand_state, time(NULL));

    mpz_ui_pow_ui(val, 10, 70); //valに1 * 10^70を代入
    mpz_add_ui(val, val, 1);

    const unsigned long long num = 500000000; //5 * 10^8を代入
    int prime_count = 0; //素数の数をカウント
    int hanpuku = 10; //MRアルゴリズムの試行回数

    for (unsigned long long i = 0; i < num; i++) {
        if (MR_test(val, hanpuku, rand_state)) { //MRアルゴリズムの結果が1(素数)の場合、素数カウントを1増やす
            prime_count++;
        }
        mpz_add_ui(val, val, 2); //valにval+2を代入
        /* 1000000ごとに現在の値を表示 */
        if ((i+1) % 1000000 == 0) {
            gmp_printf("Checked %llu numbers. Current: %Zd\n", i+1, val);
        }
    }

    printf("Total probable primes: %d\n", prime_count);

    /* メモリの解放 */
    mpz_clear(val);
    gmp_randclear(rand_state);

    end = clock();

    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %.2f seconds\n", elapsed_time);

    return 0;
}

