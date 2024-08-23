package main

import (
	_ "embed"
	"flag"
	"fmt"
	"image"
	"image/png"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/esimov/pigo/core"
)

//go:embed cascade/facefinder
var faceFinder []byte

func main() {
	var (
		out   string
		size  int
		face  float64
		shift float64
		scale float64
		angle float64
		iou   float64
		score float64
	)
	flag.StringVar(&out, "out", "faces", "output directory")
	flag.IntVar(&size, "size", 256, "minimum size of each output image")
	flag.Float64Var(&face, "face", 0.5, "face factor between 0 and 1. how much space the face occupies in the output image")
	flag.Float64Var(&shift, "shift", 0.1, "shift factor")
	flag.Float64Var(&scale, "scale", 1.1, "scale factor")
	flag.Float64Var(&angle, "angle", 0, "cascade rotation angle. 0.0 is 0 radians and 1.0 is 2*pi radians")
	flag.Float64Var(&iou, "iou", 0.2, "intersection over union threshold")
	flag.Float64Var(&score, "score", 0.5, "minimum score")
	flag.Parse()

	p := pigo.NewPigo()
	classifier, err := p.Unpack(faceFinder)
	if err != nil {
		slog.Error("failed to unpack face finder", "err", err)
		return
	}

	_ = os.MkdirAll(out, 0755)

	var wg sync.WaitGroup
	for _, arg := range flag.Args() {
		wg.Add(1)
		go func() {
			defer wg.Done()

			base := strings.TrimSuffix(filepath.Base(arg), filepath.Ext(arg))
			src, err := pigo.GetImage(arg)
			if err != nil {
				slog.Error("failed to get image", "file", arg)
				return
			}

			pixels := pigo.RgbToGrayscale(src)
			cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

			params := pigo.CascadeParams{
				MinSize:     int(float64(size) * face),
				MaxSize:     min(cols, rows),
				ShiftFactor: shift,
				ScaleFactor: scale,

				ImageParams: pigo.ImageParams{
					Pixels: pixels,
					Rows:   rows,
					Cols:   cols,
					Dim:    cols,
				},
			}

			dets := classifier.RunCascade(params, angle)
			dets = classifier.ClusterDetections(dets, iou)
			dets = slices.DeleteFunc(dets, func(d pigo.Detection) bool {
				return float64(d.Q) < score
			})

			var wg sync.WaitGroup
			for i, det := range dets {
				wg.Add(1)
				go func() {
					defer wg.Done()

					size := max(size, int(float64(det.Scale)/face))
					tl := image.Point{
						X: max(0, min(det.Col-size/2, cols-size)),
						Y: max(0, min(det.Row-size/2, rows-size)),
					}
					cr := image.Rectangle{
						Min: tl,
						Max: tl.Add(image.Point{
							X: size,
							Y: size,
						}),
					}

					slog.Info("detected face", "file", arg, "number", i, "minX", cr.Min.X, "minY", cr.Min.Y, "maxX", cr.Max.X, "maxY", cr.Max.Y)

					filename := filepath.Join(out, fmt.Sprintf("%s_%02d.png", base, i))
					crop(src, cr, filename)
				}()
				wg.Wait()
			}
		}()
	}
	wg.Wait()
}

func crop(src *image.NRGBA, cr image.Rectangle, fn string) {
	cropped := src.SubImage(cr)

	f, err := os.Create(fn)
	if err != nil {
		slog.Error("failed to create file", "file", fn, "err", err)
		return
	}
	defer func() {
		if err := f.Close(); err != nil {
			slog.Error("failed to close file", "file", fn, "err", err)
		}
	}()

	if err := png.Encode(f, cropped); err != nil {
		slog.Error("failed to encode image", "file", fn, "err", err)
	}
}
