package main

import (
    "bufio"
    "encoding/csv"
    "os"
    "fmt"
    "io"
    "strconv"
    "testing"
    "time"
)

const capacity = 1000000

// use this function as main to print the solution and clock the sigle run
func main() {

    items, weights, values, _ := get_data("nasdaq100list.csv")

    fmt.Printf("Got %d items\n", items)
    //fmt.Println(labels)
    //fmt.Println(weights)
    //fmt.Println(values)

    start := time.Now()
    solution_value, solution_weight, taken := solver(capacity, items, weights, values)
    elapsed := time.Since(start)

    fmt.Printf("\nSolution value = %d\n", solution_value)
    fmt.Printf("Solution weight = %d\n", solution_weight)
    fmt.Printf("Took %d items\n", len(taken))
    fmt.Printf("Execution time: %s", elapsed)

}

// use this function as main to run the banchmark
func main2() {

    fmt.Println(testing.Benchmark(BenchmarkSolver))

}

func get_data (filename string) (int, []int, []int, []string) {

    var weights []int
    var values []int
    var labels []string
    var items = 0

    f, _ := os.Open(filename)

    r := csv.NewReader(bufio.NewReader(f))
    for {
        record, err := r.Read()
        if err == io.EOF {
            break
        }
        labels = append(labels, record[0])
        w, err := strconv.ParseFloat(record[1], 64)
        weights = append(weights, int(w*100))
        v, err := strconv.ParseFloat(record[2], 64)
        values = append(values, int(v*100))
        items += 1
        }

    return items, weights, values, labels
}

func solver(capacity, items int, weights, values []int) (int, int, []int) {
    
    grid :=  make([][]int, items+1, items+1)
    grid[0] = make([]int, capacity+1, capacity+1) 

    for item := 0; item < items; item++ {

        grid[item+1] = make([]int, capacity+1, capacity+1)
        for k:=0; k<weights[item]; k++ {
            grid[item + 1][k] = grid[item][k]
        }
        for k:=weights[item]; k <= capacity; k++ {
            grid[item + 1][k] = max(grid[item][k], grid[item][k-weights[item]] + values[item])
        }
    } 

    solution_value := grid[items][capacity]
    solution_weight := 0
    var taken []int
    k := capacity
    for item := items; item > 0; item-- {
        if grid[item][k] != grid[item-1][k] {
            taken = append(taken, item-1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]
        }
    }

    return solution_value, solution_weight, taken
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func BenchmarkSolver(b *testing.B) {

    items, weights, values, _ := get_data("nasdaq100list.csv")
    for i := 0; i < b.N; i++ {
        solver(capacity, items, weights, values)
    }
}
