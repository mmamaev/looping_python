package main

import (
    "bufio"
    "encoding/csv"
    "os"
    "fmt"
    "io"
)

func main() {

    var weights []int
    var values []int
    var labels []string
    var items := 0
    var capacity := 1000000

    f, _ := os.Open("nasdaq100list.csv")

    r := csv.NewReader(bufio.NewReader(f))
    for {
        record, err := r.Read()
        if err == io.EOF {
            break
        }
        labels = append(labels, record[0])
        weights = append(weights, record[1])
        values = append(values, record[2])
        items += 1
        }
    }

    fmt.Println(items)
    fmt.Println(labels)


    
}