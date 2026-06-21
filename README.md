# Parallel Image Convolution for Edge Detection

This program applies a 3x3 edge-detection convolution kernel to a grayscale image.
It includes both a sequential implementation and an OpenMP parallel implementation.

## Build

```bash
make
```

## Run Sequential

```bash
./edge_detect sample.pgm output_seq.pgm seq
```

## Run OpenMP Parallel

```bash
OMP_NUM_THREADS=4 ./edge_detect sample.pgm output_omp.pgm omp
```

## Verify Correctness

```bash
./edge_detect output_seq.pgm output_omp.pgm verify
```

Expected result:

```text
VERIFY PASS: images match pixel-by-pixel.
```

## Timing / Speedup Testing

Run the same image with different thread counts:

```bash
./edge_detect sample.pgm output_seq.pgm seq
OMP_NUM_THREADS=1 ./edge_detect sample.pgm output_omp1.pgm omp
OMP_NUM_THREADS=2 ./edge_detect sample.pgm output_omp2.pgm omp
OMP_NUM_THREADS=4 ./edge_detect sample.pgm output_omp4.pgm omp
OMP_NUM_THREADS=8 ./edge_detect sample.pgm output_omp8.pgm omp
```

Speedup formula:

```text
speedup = sequential_time / parallel_time
```

## Notes

The kernel used is:

```text
-1 -1 -1
-1  8 -1
-1 -1 -1
```

This is a Laplacian-style edge detector. Each output pixel is computed independently, so the image rows are easy to divide across OpenMP threads.
