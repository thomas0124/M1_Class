const is_prime = (n) => {
    for (var i = 2; i * i <= n; i++){
        if (n % i == 0) return false
    }
    return true
}

console.log(is_prime(100000007))

10000000000000000000000000000000000000000000000000000000000000336000000