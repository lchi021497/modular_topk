# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from math import ceildiv
from sys.info import simdwidthof

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

from utils.index import Index

# ===-----------------------------------------------------------------------===#
# Naive matrix multiplication (CPU)
# ===-----------------------------------------------------------------------===#


fn naive_matrix_multiplication_cpu(
    out: ManagedTensorSlice,
    in: ManagedTensorSlice[type = out.type, rank = out.rank],
    N: out.type,
):
    """A naive matrix multiplication used as a fallback on CPU hardware."""
    var seqlen = in.shape()[0]

    for k in range(seqlen):
        var accum = 0

        for n in range(seqlen):
            accum += in[n] * (cos(2 * pi / seqlen * k * n), Complex(-sin(2 * pi / seqlen * k * n)))
        out[k] = accum 
    return out
