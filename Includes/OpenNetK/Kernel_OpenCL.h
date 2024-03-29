
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Kernel_OpenCL.h
/// \brief      (RT, OpenCL)

// CODE REVIEW  2019-07-16  KMS - Martin Dubois, ing.

#pragma once

// Macros
/////////////////////////////////////////////////////////////////////////////

#define OPEN_NET_CONSTANT             __constant
#define OPEN_NET_GLOBAL               __global
#define OPEN_NET_GLOBAL_MEMORY_FENCE  write_mem_fence(CLK_GLOBAL_MEM_FENCE)
#define OPEN_NET_GROUP_ID             get_group_id(0)
#define OPEN_NET_KERNEL               __kernel
#define OPEN_NET_PACKET_INDEX         get_local_id(0)
#define OPEN_NET_PACKET_QTY           get_local_size(0)
#define OPEN_NET_SHARED               __local
