
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Kernel_CUDA.h
/// \brief      (RT, CUDA)

// CODE REVIEW  2019-07-16  KMS - Martin Dubois, ing.

#pragma once

// Macros
/////////////////////////////////////////////////////////////////////////////

#define OPEN_NET_CONSTANT            const
#define OPEN_NET_GLOBAL
#define OPEN_NET_GLOBAL_MEMORY_FENCE __threadfence_system()
#define OPEN_NET_GROUP_ID            blockIdx.x
#define OPEN_NET_KERNEL              extern "C" __global__
#define OPEN_NET_PACKET_INDEX        threadIdx.x
#define OPEN_NET_PACKET_QTY          blockDim.x
#define OPEN_NET_SHARED              __shared__
