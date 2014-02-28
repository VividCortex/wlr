package main

/*
Copyright (c) 2014 VividCortex, Inc.  All rights reserved.
Certain inventions disclosed in this program may be claimed within
patents owned or patent applications filed by VividCortex, Inc.
*/

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
)

type csvFloater struct {
	headers []string
	*csv.Reader
}

func newCsvFloater(r io.Reader) *csvFloater {
	csvReader := csv.NewReader(r)
	return &csvFloater{
		Reader: csvReader,
	}
}

func (r *csvFloater) Read() (map[string]float64, error) {
	var (
		line []string
		row  = map[string]float64{}
		err  error
	)

	if r.headers == nil {
		r.headers, err = r.Reader.Read()
		if err != nil {
			return nil, err
		}
	}
	line, err = r.Reader.Read()
	if err != nil {
		return nil, err
	} else {
		var f float64
		for i, str := range line {
			f, err = strconv.ParseFloat(str, 64)
			if err != nil {
				return nil, err
			}
			row[r.headers[i]] = f
		}
	}
	return row, nil
}

type MultiSimple struct {
	Vars map[string]*Simple
}

func (r *MultiSimple) Add(xvalues map[string]float64, yvalue float64, verbose bool) {
	if yvalue == 0.0 {
		return
	}

	sum := 0.0
	for _, xvalue := range xvalues {
		sum += xvalue
	}
	if sum == 0.0 {
		return
	}
	slope := yvalue / sum

	for name, xValue := range xvalues {
		if xValue > 0 {
			v, present := r.Vars[name]
			if !present {
				v = &Simple{}
				r.Vars[name] = v
			}
			if verbose {
				fmt.Printf("TRAIN %s %.5g %.5g %.5g %.5g\n", name, xValue, yvalue, slope, slope*xValue)
			}
			v.Add(xValue, slope*xValue)
		}
	}
}

func (r *MultiSimple) Predict(xvalues map[string]float64) float64 {
	result := 0.0
	for name, xValue := range xvalues {
		if xValue > 0 {
			if v, present := r.Vars[name]; present {
				slope, intercept := v.Slope(), v.Intercept()
				if slope > 0 {
					if intercept < 0 {
						intercept = 0
					}
					result += intercept + xValue*slope
				}
			}
		}
	}
	return result
}

type Simple struct {
	n, sx, sy, sxx, sxy, syy float64
}

func (r *Simple) Add(x, y float64) {
	r.n++
	r.sx += x
	r.sy += y
	r.sxx += x * x
	r.sxy += x * y
	r.syy += y * y
}

func (r *Simple) Count() float64 {
	return r.n
}

func (r *Simple) Slope() float64 {
	if r.n == 0 {
		return 0
	} else if r.n == 1 {
		return r.sy / r.sx
	}
	ss_xy := r.n*r.sxy - r.sx*r.sy
	ss_xx := r.n*r.sxx - r.sx*r.sx
	return ss_xy / ss_xx
}

func (r *Simple) Intercept() float64 {
	if r.n < 2 {
		return 0
	}
	return (r.sy - r.Slope()*r.sx) / r.n
}

func (r *Simple) Rsq() float64 {
	if r.n < 2 {
		return 0
	}
	ss_xy := r.n*r.sxy - r.sx*r.sy
	ss_xx := r.n*r.sxx - r.sx*r.sx
	ss_yy := r.n*r.syy - r.sy*r.sy
	return ss_xy * ss_xy / ss_xx / ss_yy
}

func (r *Simple) SlopeStderr() float64 {
	if r.n <= 2 {
		return 0
	}
	ss_xy := r.n*r.sxy - r.sx*r.sy
	ss_xx := r.n*r.sxx - r.sx*r.sx
	ss_yy := r.n*r.syy - r.sy*r.sy
	s := math.Sqrt((ss_yy - ss_xy*ss_xy/ss_xx) / (r.n - 2.0))
	return s / math.Sqrt(ss_xx)
}

func (r *Simple) InterceptStderr() float64 {
	if r.n <= 2 {
		return 0
	}
	ss_xy := r.n*r.sxy - r.sx*r.sy
	ss_xx := r.n*r.sxx - r.sx*r.sx
	ss_yy := r.n*r.syy - r.sy*r.sy
	s := math.Sqrt((ss_yy - ss_xy*ss_xy/ss_xx) / (r.n - 2.0))
	mean_x := r.sx / r.n
	return s * math.Sqrt(1.0/r.n+mean_x*mean_x/ss_xx)
}

func main() {
	var (
		train, predict string
		yvar           = "user_us"
		ms             = MultiSimple{
			Vars: map[string]*Simple{},
		}
		r                 = Simple{}
		totalError, count float64
	)

	if len(os.Args) < 2 || len(os.Args) > 3 {
		log.Fatalln("Usage: go run regress.go <file to train on> [<file to predict>]")
	}
	train = os.Args[1]
	predict = train
	if len(os.Args) == 3 {
		predict = os.Args[2]
	}

	tfh, err := os.Open(train)
	if err != nil {
		log.Fatalln(err)
	}
	defer tfh.Close()
	pfh, err := os.Open(predict)
	if err != nil {
		log.Fatal(err)
	}
	defer pfh.Close()
	c := newCsvFloater(tfh)
	c2 := newCsvFloater(pfh)

	// train
	fmt.Println("TRAIN name xValue yValue slope contrib")
	for {
		row, err := c.Read()
		if err != nil {
			if err == io.EOF {
				break
			} else {
				log.Fatalln(err)
			}
		}
		yvalue := row[yvar]
		delete(row, yvar)
		ms.Add(row, yvalue, true)
	}

	// predict
	fmt.Println("PREDICT actual predicted")
	for {
		row, err := c2.Read()
		if err != nil {
			if err == io.EOF {
				break
			} else {
				log.Fatalln(err)
			}
		}
		yvalue := row[yvar]
		if yvalue != 0 {
			delete(row, yvar)
			pred := ms.Predict(row)
			count++
			totalError += math.Abs((yvalue - pred) / yvalue)
			r.Add(yvalue, pred)
			fmt.Printf("PREDICT %.5g %.5g\n", yvalue, pred)
		}
	}

	fmt.Println()
	fmt.Println("================== RESULTS: VARIABLES ====================")
	fmt.Println()
	fmt.Println("variable             count  R^2     slope  (t-stat) intercept  (t-stat)")
	for name, v := range ms.Vars {
		fmt.Printf("%-20s %5.f %4.2f %9.3g %9.3g %9.3g %9.3g\n", name, v.Count(), v.Rsq(),
			v.Slope(), v.SlopeStderr()/v.Slope(), v.Intercept(), math.Abs(v.InterceptStderr()/v.Intercept()))
	}

	fmt.Println()
	fmt.Println("================= RESULTS: ACTUAL-VS-PRED ===================")
	fmt.Printf("Slope: %.2g T-stat: %.2g Intercept: %.2g T-stat: %.2g R^2 %.2g MAPE: %.2g\n",
		r.Slope(), r.SlopeStderr()/r.Slope(), r.Intercept(), math.Abs(r.InterceptStderr()/r.Intercept()),
		r.Rsq(), totalError/count)
}
