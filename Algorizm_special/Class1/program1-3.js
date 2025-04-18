const MergeSort =(a) => {
    const Merge = (a, b) => {
        let res = []
        while (true) {
            if (a.length == 0) return res.concat(b)
            else if (b.length == 0) return res.concat(a)
            else{
                if (a[0] < b[0]) res.push(a.shift())
                else res.push(b.shift())
            }
        }
    }
    let n = a.length;
    if (n <= 1) return a
    else {
        let n2 = Math.floor(n / 2);
        return Merge(MergeSort(a.slice(0, n2)), MergeSort(a.slice(n2, n)))
    }
}

console.log(MergeSort([7, 6, 4, 5, 3, 1, 2, 7, 1]))