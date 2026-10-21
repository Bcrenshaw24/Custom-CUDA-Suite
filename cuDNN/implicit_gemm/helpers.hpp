#pragma once


/**
 * @brief Calculates output dimensions for Implicit GEMM
 * 
 *
 * @param H Input height. 
 * @param W Input width. 
 * @param R Filter height.  
 * @param S Filter width.   
 * @param P Output height.   
 * @param Q Output width.   
 * @param pad_t Top padding.   
 * @param pad_b Bottom padding.   
 * @param pad_l Left padding.   
 * @param pad_r Right padding.   
 * @param stride_h Vertical stride.
 * @param stride_w Horizontal stride. 
 * @param dial_h Vertical dialation.
 * @param dial_w Horizontal dialation.
 * 
 */
void getOutputDimensions(
    int H, int W, 
    int R, int S, 
    int& P, int& Q,
    int pad_t=1, int pad_b=1, int pad_l=1, int pad_r=1, 
    int stride_h=1, int stride_w=1,
    int dial_h=1, int dial_w=1
);