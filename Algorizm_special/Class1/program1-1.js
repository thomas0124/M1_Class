const sum = (a) => {
    let n = a.length;
    var s = 0;
    for (let i = 0; i < n; i++) {
        s += a[i];
    }
    return s;
}

console.log(sum([1, 2, 3, 4, 5]));