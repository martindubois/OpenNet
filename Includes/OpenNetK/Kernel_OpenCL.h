
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Kernel_OpenCL.h
/// \todo       Document macros

#pragma once

// Macros
/////////////////////////////////////////////////////////////////////////////

#define OPEN_NET_DEVICE
#define OPEN_NET_GLOBAL               __global
#define OPEN_NET_GLOBAL_MEMORY_FENCE  write_mem_fence(CLK_GLOBAL_MEM_FENCE)
#define OPEN_NET_KERNEL               __kernel
#define OPEN_NET_PACKET_INDEX         get_local_id(0)
