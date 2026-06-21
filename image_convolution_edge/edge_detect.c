/*
 * edge_detect.c
 * CSCI 551 Final Parallel Program
 *
 * Program: Sequential and OpenMP image convolution for edge detection.
 * Image format: ASCII PGM (P2) grayscale images.
 *
 * Features:
 *   - Sequential convolution mode
 *   - OpenMP convolution mode
 *   - Pixel-by-pixel verification mode
 *   - Contrast normalization for clearer edge-detection output
 *
 * Build:
 *   make
 *
 * Run sequential:
 *   ./edge_detect input.pgm output_seq.pgm seq
 *
 * Run OpenMP parallel:
 *   OMP_NUM_THREADS=4 ./edge_detect input.pgm output_omp.pgm omp
 *
 * Verify two output images:
 *   ./edge_detect output_seq.pgm output_omp.pgm verify
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define KERNEL_SIZE 3

/* Laplacian-style edge detection kernel.
 * Center pixel is weighted heavily and neighboring pixels are subtracted.
 * Strong intensity changes become edges.
 */
static const int EDGE_KERNEL[KERNEL_SIZE][KERNEL_SIZE] = {
    {-1, -1, -1},
    {-1,  8, -1},
    {-1, -1, -1}
};

typedef struct {
    int width;
    int height;
    int maxval;
    int *pixels;   /* int allows raw convolution values before normalization */
} Image;

static double seconds_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int clamp_int(int value, int low, int high)
{
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static void free_image(Image *img)
{
    if (img != NULL) {
        free(img->pixels);
        img->pixels = NULL;
    }
}

/* Read a simple ASCII PGM P2 file.
 * This reader supports comment lines beginning with '#'.
 */
static int read_pgm(const char *filename, Image *img)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening input file");
        return 0;
    }

    char magic[3];

    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P2") != 0) {
        fprintf(stderr, "Error: only ASCII PGM P2 files are supported.\n");
        fclose(fp);
        return 0;
    }

    /* Skip whitespace and comment lines before width/height. */
    int c = fgetc(fp);
    while (c != EOF) {
        while (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            c = fgetc(fp);
        }

        if (c == '#') {
            while (c != '\n' && c != EOF) {
                c = fgetc(fp);
            }
            c = fgetc(fp);
        } else {
            break;
        }
    }

    if (c != EOF) {
        ungetc(c, fp);
    }

    if (fscanf(fp, "%d %d", &img->width, &img->height) != 2) {
        fprintf(stderr, "Error: could not read image width/height.\n");
        fclose(fp);
        return 0;
    }

    if (fscanf(fp, "%d", &img->maxval) != 1 || img->maxval <= 0 || img->maxval > 255) {
        fprintf(stderr, "Error: maxval must be between 1 and 255.\n");
        fclose(fp);
        return 0;
    }

    size_t total = (size_t)img->width * (size_t)img->height;

    img->pixels = (int *)malloc(total * sizeof(int));
    if (img->pixels == NULL) {
        fprintf(stderr, "Error: malloc failed for image pixels.\n");
        fclose(fp);
        return 0;
    }

    for (size_t i = 0; i < total; i++) {
        int value;

        if (fscanf(fp, "%d", &value) != 1) {
            fprintf(stderr, "Error: not enough pixel data in file.\n");
            free_image(img);
            fclose(fp);
            return 0;
        }

        img->pixels[i] = clamp_int(value, 0, img->maxval);
    }

    fclose(fp);
    return 1;
}

static int write_pgm(const char *filename, const Image *img)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Error opening output file");
        return 0;
    }

    /* No comment line here, so this output can be read again by strict PGM readers. */
    fprintf(fp, "P2\n");
    fprintf(fp, "%d %d\n", img->width, img->height);
    fprintf(fp, "%d\n", img->maxval);

    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            int val = clamp_int(img->pixels[y * img->width + x], 0, img->maxval);
            fprintf(fp, "%d ", val);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 1;
}

/* Normalize raw convolution output to 0-255 for better visual contrast.
 * This should be called after the convolution is complete.
 */
static void normalize_image(Image *img)
{
    size_t total = (size_t)img->width * (size_t)img->height;

    int min = img->pixels[0];
    int max = img->pixels[0];

    for (size_t i = 1; i < total; i++) {
        if (img->pixels[i] < min) min = img->pixels[i];
        if (img->pixels[i] > max) max = img->pixels[i];
    }

    if (max == min) {
        for (size_t i = 0; i < total; i++) {
            img->pixels[i] = 0;
        }
        return;
    }

    for (size_t i = 0; i < total; i++) {
        img->pixels[i] = (img->pixels[i] - min) * img->maxval / (max - min);
    }
}

/* Sequential 3x3 convolution. */
static void edge_convolution_seq(const Image *input, Image *output)
{
    int w = input->width;
    int h = input->height;

    output->width = w;
    output->height = h;
    output->maxval = input->maxval;

    size_t total = (size_t)w * (size_t)h;

    output->pixels = (int *)calloc(total, sizeof(int));
    if (output->pixels == NULL) {
        fprintf(stderr, "Error: calloc failed for output image.\n");
        exit(EXIT_FAILURE);
    }

    /* Keep border pixels black and convolve only interior pixels. */
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int sum = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = input->pixels[(y + ky) * w + (x + kx)];
                    int weight = EDGE_KERNEL[ky + 1][kx + 1];

                    sum += pixel * weight;
                }
            }

            /* Store raw value first. Do not clamp here. */
            output->pixels[y * w + x] = sum;
        }
    }

    normalize_image(output);
}

/* OpenMP parallel version.
 * Each output pixel is independent, so rows can be divided among threads.
 */
static void edge_convolution_omp(const Image *input, Image *output)
{
    int w = input->width;
    int h = input->height;

    output->width = w;
    output->height = h;
    output->maxval = input->maxval;

    size_t total = (size_t)w * (size_t)h;

    output->pixels = (int *)calloc(total, sizeof(int));
    if (output->pixels == NULL) {
        fprintf(stderr, "Error: calloc failed for output image.\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for schedule(static)
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int sum = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = input->pixels[(y + ky) * w + (x + kx)];
                    int weight = EDGE_KERNEL[ky + 1][kx + 1];

                    sum += pixel * weight;
                }
            }

            /* Store raw value first. Do not clamp here. */
            output->pixels[y * w + x] = sum;
        }
    }

    /* Normalize after all threads finish computing raw values. */
    normalize_image(output);
}

static int verify_images(const Image *a, const Image *b)
{
    if (a->width != b->width || a->height != b->height || a->maxval != b->maxval) {
        printf("VERIFY FAIL: image dimensions or maxval do not match.\n");
        return 0;
    }

    size_t total = (size_t)a->width * (size_t)a->height;
    size_t mismatches = 0;

    for (size_t i = 0; i < total; i++) {
        if (a->pixels[i] != b->pixels[i]) {
            mismatches++;

            if (mismatches <= 10) {
                printf("Mismatch at pixel %zu: %d vs %d\n", i, a->pixels[i], b->pixels[i]);
            }
        }
    }

    if (mismatches == 0) {
        printf("VERIFY PASS: images match pixel-by-pixel.\n");
        return 1;
    }

    printf("VERIFY FAIL: %zu mismatched pixels.\n", mismatches);
    return 0;
}

static void print_usage(const char *program)
{
    printf("Usage:\n");
    printf("  %s input.pgm output.pgm seq\n", program);
    printf("  %s input.pgm output.pgm omp\n", program);
    printf("  %s image_a.pgm image_b.pgm verify\n", program);
}

int main(int argc, char *argv[])
{
    if (argc != 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    const char *mode = argv[3];

    if (strcmp(mode, "verify") == 0) {
        Image a = {0, 0, 0, NULL};
        Image b = {0, 0, 0, NULL};

        if (!read_pgm(input_file, &a) || !read_pgm(output_file, &b)) {
            free_image(&a);
            free_image(&b);
            return EXIT_FAILURE;
        }

        int ok = verify_images(&a, &b);

        free_image(&a);
        free_image(&b);

        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    Image input = {0, 0, 0, NULL};
    Image output = {0, 0, 0, NULL};

    if (!read_pgm(input_file, &input)) {
        return EXIT_FAILURE;
    }

    double start = seconds_now();

    if (strcmp(mode, "seq") == 0) {
        edge_convolution_seq(&input, &output);
    } else if (strcmp(mode, "omp") == 0) {
        edge_convolution_omp(&input, &output);
    } else {
        fprintf(stderr, "Error: mode must be seq, omp, or verify.\n");
        free_image(&input);
        return EXIT_FAILURE;
    }

    double end = seconds_now();

    if (!write_pgm(output_file, &output)) {
        free_image(&input);
        free_image(&output);
        return EXIT_FAILURE;
    }

#ifdef _OPENMP
    int threads = omp_get_max_threads();
#else
    int threads = 1;
#endif

    printf("Mode: %s\n", mode);
    printf("Image: %d x %d\n", input.width, input.height);
    printf("Threads available: %d\n", threads);
    printf("clock_gettime elapsed = %.9f sec\n", end - start);
    printf("Output written to: %s\n", output_file);

    free_image(&input);
    free_image(&output);

    return EXIT_SUCCESS;
}
