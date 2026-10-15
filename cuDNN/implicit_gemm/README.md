# Implicit GEMM 

## Algorithm Overview & Considerations

- A 3D convolutional kernel slides over an input matrix (like an image) and produces a single weighted sum as an output. 

A few considerations as for how this works: 

- A simple kernel by itself reduces output dimensions 
- Edge detection filters do NOT reduce the output dimensions 
- Tiles that are not divisible by kernel size need padding 
- A third thread axis is needed to account for channels/kernels to avoid repeated computation 

I made the first two points to display that not all uses of this kernel will result in smaller matricies meaning that we may need multiple launch implementations. 

The second point describes how a kernel may step over the tiled blocks and need data from another which is restricted causing them to fetch from global memory. We can prevent this by padding the tile matrix with our kernel size.

Lastly, CUDA allows for an x, y, and z axis to account for multi-dimensional blocks of data though z is more limited than x and y. 

## Parameters

Convolutions are comprised of several paramters that affect the kernel: 

- Input Size: Dimensions of Input Matrix
- Kernel Size: Dimensions of Weight Matrix 
- Padding: Amount of padded space on Input Matrix 
- Dialation: Offset added to Weight Matrix threads
- Num of Filters: Number of Weight Matricies (Typically equal to # of input channels)

The kernel doesn't concern itself with padding from what I can tell as this is a preprocessing step performed by the launch code. 

Kernel size seems to typically have square dimensions but rectangular ones are valid so that will be added. 

## Kernel Considerations

