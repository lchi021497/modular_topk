from gpu import WARP_SIZE, barrier, block_idx, thread_idx, warp_id
from gpu.host import DeviceBuffer, DeviceContext
from gpu.memory import async_copy_wait_all
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
)
from layout.math import outer_product_acc
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import TensorCore
from memory import UnsafePointer
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor, foreach
from math import cos, sin, pi
from utils.index import Index
from complex import ComplexSIMD
from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator

# ===-----------------------------------------------------------------------===#
# FFT Implementation
# ===-----------------------------------------------------------------------===#

@value
struct FFT:
    """Fast Fourier Transform implementation."""
    
    @staticmethod
    fn naive_cpu(
        out: ManagedTensorSlice,
        input_tensor: ManagedTensorSlice[type = out.type, rank = out.rank],
    ) -> ManagedTensorSlice:
        """A naive FFT implementation used as a fallback on CPU hardware."""
        var seqlen = input_tensor.shape()[0]

        for k in range(seqlen):
            var accum = Complex[out.type](0, 0)

            for n in range(seqlen):
                var angle = 2 * pi * k * n / seqlen
                var twiddle = Complex[out.type](cos(angle), -sin(angle))
                accum += input_tensor[n] * twiddle
                
            out[k] = accum 
        return out
    
    @staticmethod
    fn fft_gpu(
        ctx: DeviceContextPtr,
        input_tensor: ManagedTensorSlice,
        output_tensor: ManagedTensorSlice
    ):
        """GPU-accelerated FFT implementation."""
        
        # Check if we have a supported GPU
        var has_gpu = has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
        if not has_gpu:
            # Fallback to CPU if no GPU is available
            _ = FFT.naive_cpu(output_tensor, input_tensor)
            return
            
        var seqlen = input_tensor.shape()[0]
        
        # Get a device context for the GPU
        var device_ctx = DeviceContext(ctx._ptr)
        
        # Create device buffers for input and output
        var input_buffer = device_ctx.enqueue_create_buffer[input_tensor.type](seqlen * 2)  # *2 for complex numbers
        var output_buffer = device_ctx.enqueue_create_buffer[output_tensor.type](seqlen * 2)
        
        # Define layouts
        alias layout = Layout.row_major(seqlen * 2)
        
        # Copy input data to device
        with input_buffer.map_to_host() as host_buffer:
            var host_tensor = LayoutTensor[input_tensor.type, layout](host_buffer)
            for i in range(seqlen):
                var complex_val = input_tensor[i]
                host_tensor[i*2] = complex_val.real()      # Real part
                host_tensor[i*2 + 1] = complex_val.imag()  # Imaginary part
        
        # Wrap device buffers in tensors
        var device_input = LayoutTensor[input_tensor.type, layout](input_buffer)
        var device_output = LayoutTensor[output_tensor.type, layout](output_buffer)
        
        # Launch GPU kernel
        device_ctx.enqueue_function[fft_kernel](
            device_input,
            device_output,
            seqlen,
            grid_dim=1,
            block_dim=seqlen  # One thread per output element
        )
        
        # Copy results back to host
        with output_buffer.map_to_host() as host_buffer:
            var host_tensor = LayoutTensor[output_tensor.type, layout](host_buffer)
            for i in range(seqlen):
                var real = host_tensor[i*2]
                var imag = host_tensor[i*2 + 1]
                output_tensor[i] = Complex[output_tensor.type](real, imag)
    
    @staticmethod
    fn fft_forward(
        ctx: DeviceContextPtr,
        input: InputTensor,
        output: OutputTensor
    ):
        """Forward FFT computation."""
        # Get tensor slices from input/output tensors
        var input_slice = input.tensor()
        var output_slice = output.tensor()
        
        # Use GPU implementation if available, otherwise fall back to CPU
        if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
            FFT.fft_gpu(ctx, input_slice, output_slice)
        else:
            # Call naive CPU implementation as fallback
            FFT.naive_cpu(output_slice, input_slice)

# GPU kernel for FFT
fn fft_kernel(
    input_tensor: LayoutTensor[DType.float32, Layout.any(), MutableAnyOrigin],
    output_tensor: LayoutTensor[DType.float32, Layout.any(), MutableAnyOrigin],
    seqlen: Int
):
    """GPU kernel for Fast Fourier Transform."""
    var k = thread_idx.x  # One thread per output element
    
    if k < seqlen:
        var real_sum: DType.float32 = 0.0
        var imag_sum: DType.float32 = 0.0
        
        for n in range(seqlen):
            var angle = -2.0 * pi * DType.float32(k * n) / DType.float32(seqlen)
            var cos_val = cos(angle)
            var sin_val = sin(angle)
            
            # Get input complex value
            var in_real = input_tensor[n*2]      # Real part
            var in_imag = input_tensor[n*2 + 1]  # Imaginary part
            
            # Complex multiplication
            real_sum += in_real * cos_val - in_imag * sin_val
            imag_sum += in_real * sin_val + in_imag * cos_val
        
        # Store result
        output_tensor[k*2] = real_sum      # Real part
        output_tensor[k*2 + 1] = imag_sum  # Imaginary part

# Register operation with the runtime
fn register_fft_ops():
    register_op("FFT", FFT.fft_forward)