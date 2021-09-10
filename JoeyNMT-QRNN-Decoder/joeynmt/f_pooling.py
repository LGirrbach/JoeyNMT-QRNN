import math
import torch
import cupy as cp
from cupy.cuda import function

###

kernel = '''
extern "C"
__global__ void recurrent_forget_mult(float *dst, const float *f, const float *z, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  for (int ts = 0 + 1; ts < SEQ + 1; ts++) {
     // Good sanity check for debugging - only perform additions to a zeroed chunk of memory
     // Addition seems atomic or near atomic - you should get incorrect answers if doubling up via threads
     // Note: the index i needs to be offset by one as f[0] (f_t) is used for dst[1] (h_t) etc

     // To move timesteps, we step HIDDEN * BATCH
     // To move batches, we move HIDDEN
     // To move neurons, we move +- 1
     // Note: dst[dst_i] = ts * 100 + bid * 10 + hid; is useful for debugging

     int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_i       = (ts - 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     dst[dst_i]      = (1 - f[i]) * z[i];
     dst[dst_i]      += f[i] * dst[dst_iminus1];
  }
}

extern "C"
__global__ void bwd_recurrent_forget_mult(const float *h, const float *f, const float *z, const float *gh, float *gf, float *gz, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  double running_f = 0;
  for (int ts = SEQ - 1 + 1; ts >= 0 + 1; ts--) {
     int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     // int dst_i       = (ts - 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     //
     running_f       += gh[dst_iminus1];
     // Gradient of X
     gz[i]           = (1 - f[i]) * running_f;
     // Gradient of F
     gf[i]           = (h[dst_iminus1] - z[i]) * running_f;
     //
     // The line below is likely more numerically stable than (1 - f[i]) * running_f;
     running_f       = f[i] * running_f;
  }
}
'''

###

class CPUForgetMult(torch.nn.Module):
    def __init__(self):
        super(CPUForgetMult, self).__init__()

    def forward(self, f, z):
        result = []
        ###
        forgets = f.split(1, dim=0)
        prev_h = None
        for i, h in enumerate(((1-f) * z).split(1, dim=0)):
            if prev_h is not None: h = h + forgets[i] * prev_h
            # h is (1, batch, hidden) when it needs to be (batch_hidden)
            # Calling squeeze will result in badness if batch size is 1
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h
        ###
        return torch.stack(result)


class GPUForgetMult(torch.autograd.Function):
    module = cp.RawModule(code=kernel)
    forget_mult = module.get_function('recurrent_forget_mult')
    forget_mult_backward = module.get_function('bwd_recurrent_forget_mult')

    @staticmethod
    def forward(ctx, f, z):
        # self.compile()
        seq_size, batch_size, hidden_size = f.size()
        result = torch.zeros(seq_size + 1, batch_size, hidden_size)
        # We only zero the result array (result[0]) if we don't set a hidden initial state
        # All other values (result[1:]) are overwritten by default
        ###
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)
        GPUForgetMult.forget_mult(grid=grid,
                                  block=(grid_hidden_size, 1),
                                  args=[result.data_ptr(), f.data_ptr(), z.data_ptr(), seq_size, batch_size, hidden_size])
        ctx.save_for_backward(f, z, result)
        return result[1:, :, :]
    
    @staticmethod
    def backward(ctx, grad_h):
        # self.compile()
        f, z, h = ctx.saved_tensors
        grad_h = grad_h.contiguous()
        ###
        seq_size, batch_size, hidden_size = f.size()
        # Zeroing is not necessary as these will be overwritten
        grad_f = f.new(*f.size())
        grad_z = f.new(*f.size())
        ###
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)
        GPUForgetMult.forget_mult_backward(grid=grid,
                                  block=(grid_hidden_size, 1),
                                  args=[h.data_ptr(), f.data_ptr(),
                                        z.data_ptr(), grad_h.data_ptr(),
                                        grad_f.data_ptr(), grad_z.data_ptr(),
                                        seq_size, batch_size,
                                        hidden_size])
        ###
        return grad_f, grad_z


class FPooling(torch.nn.Module):
    r"""ForgetMult computes a simple recurrent equation:
    h_t = f_t * x_t + (1 - f_t) * h_{t-1}

    This equation is equivalent to dynamic weighted averaging.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - F (seq_len, batch, input_size): tensor containing the forget gate values, assumed in range [0, 1].
        - hidden_init (batch, input_size): tensor containing the initial hidden state for the recurrence (h_{t-1}).
        - use_cuda: If True, use the fast element-wise CUDA kernel for recurrence. If False, uses naive for loop. Default: True.
    """

    def __init__(self):
        super(FPooling, self).__init__()
        self.gpu_forget_mult = GPUForgetMult.apply
        self.cpu_forget_mult = CPUForgetMult()

    def forward(self, f, z):
        # Use CUDA by default unless it's available
        use_cuda = f.is_cuda and z.is_cuda

        z = z.permute(2, 0, 1).contiguous()
        f = f.permute(2, 0, 1).contiguous()

        if use_cuda:
            return self.gpu_forget_mult(f, z).permute(1, 2, 0)
        else:
            return self.cpu_forget_mult(f, z).permute(1, 2, 0)
