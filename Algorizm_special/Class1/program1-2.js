const BubbleSort = (a) => {
    let n = a.length;
    for (let i = n; i > 0; i--){
        for (let j = 0; j < i - 1; j++){
            if (a[j] > a[j + 1]) [a[j], a[j + 1]] = [a[j + 1], a[j]]
        }
    }
    return a
}

console.log(BubbleSort([5, 4, 3, 2, 1]));