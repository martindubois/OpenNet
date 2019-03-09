
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Kernel_CUDA.h
/// \todo       Document macros

#pragma once

// Macros
/////////////////////////////////////////////////////////////////////////////

#define OPEN_NET_DEVICE              __device__
#define OPEN_NET_GLOBAL
#define OPEN_NET_GLOBAL_MEMORY_FENCE __threadfence_system()
#define OPEN_NET_GROUP_ID            blockIdx.x
#define OPEN_NET_KERNEL              extern "C" __global__
#define OPEN_NET_PACKET_INDEX        threadIdx.x
#define OPEN_NET_PACKET_QTY          blockDim.x
