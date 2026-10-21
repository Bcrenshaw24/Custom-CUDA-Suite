#include "helpers.hpp"


void getOutputDimensions(
    int H, int W, 
    int R, int S, 
    int& P, int& Q,
    int pad_t, int pad_b, int pad_l, int pad_r, 
    int stride_w, int stride_h,
    int dial_h, int dial_w
)
{ 
    P = (((W + pad_l + pad_r) - dial_w * (S - 1) - 1) / stride_w) + 1;
    Q = (((H + pad_t + pad_b) - dial_h * (R - 1) - 1) / stride_h) + 1;
}