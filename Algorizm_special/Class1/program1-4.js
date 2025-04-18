const NondeterministicSorter = (lst) => {
    let lst1 =[]
    let i, j;
    let lst2 = lst.length;
    for (i = 0; i < lst2; i++){
        j = choice(0, n)
        if (lst1[j] == undefined){
            lst1[j] = lst[i]
        }
        else failure()
    }
    for (i = 0; i < n - 1; i++){
        if (lst1[i] > lst1[i + 1]){
            failure()
        }
    }
    return lst1
}
console.log(NondeterministicSorter([7, 6, 4, 5, 3, 1, 2, 7, 1]))