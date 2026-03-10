#include <math.h>

#include "fourier.h"

void normalize(complex s[], int n) {
    for (int k = 0; k < n; k++) {
        s[k].a /= n;
        s[k].b /= n;
    }
}

void nft(complex s[], complex t[], int n, int sign) {
    for (int k = 0; k < n; k++) {
        t[k].a = 0;
        t[k].b = 0;

        for (int j = 0; j < n; j++) {
            double x = sign * 2 * PI * k * j / n;

            double cosx = cos(x);
            double sinx = sin(x);

            t[k].a += s[j].a * cosx - s[j].b * sinx;
            t[k].b += s[j].a * sinx + s[j].b * cosx;
        }
    }
}

void nft_forward(complex s[], complex t[], int n) {
    nft(s, t, n, -1);
}

void nft_inverse(complex t[], complex s[], int n) {
    nft(t, s, n, 1);
    normalize(s, n);
}

void fft(complex s[], complex t[], int n, int sign) {
    if (n == 1) {
        t[0] = s[0];
        return;
    }

    int metade = n / 2;
    complex sp[metade], si[metade];
    for (int k = 0; k < metade; k++) {
        sp[k] = s[2 * k];
        si[k] = s[2 * k + 1];
    }

    complex tp[metade], ti[metade];
    fft(sp, tp, metade, sign);
    fft(si, ti, metade, sign);

    for (int k = 0; k < metade; k++) {
        double x = sign * 2 * PI * k / n;
        double cosx = cos(x);
        double sinx = sin(x);

        // w = e**(sinal * 2πki/n) * ti[k]
        complex w;
        w.a = ti[k].a * cosx - ti[k].b * sinx;
        w.b = ti[k].a * sinx + ti[k].b * cosx;

        t[k].a = tp[k].a + w.a;
        t[k].b = tp[k].b + w.b;

        t[k + metade].a = tp[k].a - w.a;
        t[k + metade].b = tp[k].b - w.b;
    }

}

void fft_forward(complex s[], complex t[], int n) {
    fft(s, t, n, -1);
}

void fft_inverse(complex t[], complex s[], int n) {
    fft(t, s, n, 1);
    normalize(s, n);
}

void fft_forward_2d(complex matrix[MAX_SIZE][MAX_SIZE], int width, int height) {
    for (int y = 0; y < width; y++) {
        complex s[height], t[height];
        for (int x = 0; x < height; x++) {
            s[x] = matrix[x][y];
        }

        fft_forward(s, t, height);

        for (int x = 0; x < height; x++) {
            matrix[x][y] = t[x];
        }
    }

    for (int x = 0; x < height; x++) {
        complex s[width], t[width];
        for (int y = 0; y < width; y++) {
            s[y] = matrix[x][y];
        }

        fft_forward(s, t, width);

        for (int y = 0; y < width; y++) {
            matrix[x][y] = t[y];
        }
    }
}

void fft_inverse_2d(complex matrix[MAX_SIZE][MAX_SIZE], int width, int height) {
    for (int x = 0; x < height; x++) {
        complex s[width], t[width];
        for (int y = 0; y < width; y++) {
            s[y] = matrix[x][y];
        }

        fft_inverse(s, t, width);

        for (int y = 0; y < width; y++) {
            matrix[x][y] = t[y];
        }
    }

    for (int y = 0; y < width; y++) {
        complex s[height], t[height];
        for (int x = 0; x < height; x++) {
            s[x] = matrix[x][y];
        }

        fft_inverse(s, t, height);

        for (int x = 0; x < height; x++) {
            matrix[x][y] = t[x];
        }
    }
}

void filter(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height, int flip) {
    int center_x = width / 2;
    int center_y = height / 2;

    double variance = -2 * SIGMA * SIGMA;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dx = center_x - (x + center_x) % width;
            int dy = center_y - (y + center_y) % height;

            double d = dx * dx + dy * dy;
            double g = exp(d / variance);

            if (flip) {
                g = 1 - g;
            }

            output[y][x].a = g * input[y][x].a;
            output[y][x].b = g * input[y][x].b;
        }
    }
}

void filter_lp(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height) {
    filter(input, output, width, height, 0);
}

void filter_hp(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height) {
    filter(input, output, width, height, 1);
}
