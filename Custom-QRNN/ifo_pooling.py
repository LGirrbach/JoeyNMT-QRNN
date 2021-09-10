import math
import torch
import cupy as cp
from cupy.cuda import function

###

kernel = '''
extern "C"
__global__ void recurrent_forget_mult(float *dst, const float *f, const float *z, const float *i, int SEQ, int BATCH, int HIDDEN)
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

     int j           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_j       = (ts - 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_jminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     dst[dst_j]      = i[j] * z[j];
     dst[dst_j]      += f[j] * dst[dst_jminus1];
  }
}

extern "C"
__global__ void bwd_recurrent_forget_mult(const float *h, const float *f, const float *z, const float *i, const float *gh, float *gf, float *gz, float *gi, float *ghinit, int SEQ, int BATCH, int HIDDEN)
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
     int j           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_jminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     //
     running_f       += gh[dst_jminus1];
     // Gradient of X
     gz[j]           = i[j] * running_f;
     gi[j]           = z[j] * running_f;
     // Gradient of F
     gf[j]           = h[dst_jminus1] * running_f;
     //
     running_f       = f[j] * running_f;
  }
  ghinit[bid * HIDDEN + hid] = running_f;
}
'''

###

class CPUForgetMult(torch.nn.Module):
    def __init__(self):
        super(CPUForgetMult, self).__init__()

    def forward(self, f, z, i, hidden_init=None):
        result = []
        ###
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((i * z).split(1, dim=0)):
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
    def forward(ctx, f, z, i, hidden_init=None):
        # self.compile()
        seq_size, batch_size, hidden_size = f.size()
        result = f.new(seq_size + 1, batch_size, hidden_size)
        # We only zero the result array (result[0]) if we don't set a hidden initial state
        # All other values (result[1:]) are overwritten by default
        if hidden_init is not None:
            result[0, :, :] = hidden_init
        else:
            result = result.zero_()
        ###
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)
        GPUForgetMult.forget_mult(grid=grid,
                                  block=(grid_hidden_size, 1),
                                  args=[result.data_ptr(), f.data_ptr(), z.data_ptr(), i.data_ptr(),
                                        seq_size, batch_size, hidden_size])
        ctx.save_for_backward(f, z, i, hidden_init, result)
        return result[1:, :, :]
    
    @staticmethod
    def backward(ctx, grad_h):
        # self.compile()
        f, z, i, hidden_init, h = ctx.saved_tensors
        grad_h = grad_h.contiguous()
        ###
        seq_size, batch_size, hidden_size = f.size()
        # Zeroing is not necessary as these will be overwritten
        grad_f = f.new(*f.size())
        grad_z = f.new(*f.size())
        grad_i = f.new(*f.size())
        grad_h_init = f.new(batch_size, hidden_size)
        ###
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)
        GPUForgetMult.forget_mult_backward(grid=grid,
                                  block=(grid_hidden_size, 1),
                                  args=[h.data_ptr(), f.data_ptr(),
                                        z.data_ptr(), i.data_ptr(), grad_h.data_ptr(),
                                        grad_f.data_ptr(), grad_z.data_ptr(), grad_i.data_ptr(),
                                        grad_h_init.data_ptr(),
                                        seq_size, batch_size,
                                        hidden_size])
        ###
        if hidden_init is not None:
            return grad_f, grad_z, grad_i, grad_h_init
        return grad_f, grad_z, grad_i


class IFOPooling(torch.nn.Module):
    def __init__(self):
        super(IFOPooling, self).__init__()
        self.gpu_forget_mult = GPUForgetMult.apply
        self.cpu_forget_mult = CPUForgetMult()

    def forward(self, f, z, i):
        # Use CUDA by default unless it's available
        use_cuda = f.is_cuda and z.is_cuda and i.is_cuda
        # Have to make tensors contiguous after permuting
        # CUDA Kernel expects (seq, batch, features)
        # but QRNNLayer provides (batch, features, seq)
        # because of convolution
        z = z.permute(2, 0, 1).contiguous()
        f = f.permute(2, 0, 1).contiguous()
        i = i.permute(2, 0, 1).contiguous()

        if use_cuda:
            return self.gpu_forget_mult(f, z, i).permute(1, 2, 0)
        else:
            return self.cpu_forget_mult(f, z, i).permute(1, 2, 0)
