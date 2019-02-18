
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Kernel.h
/// \brief      Include the right header file based on the language

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================

#ifdef _OPEN_NET_CUDA_
    #include <OpenNetK/Kernel_CUDA.h>
#endif

#ifdef _OPEN_NET_OPEN_CL_
    #include <OpenNetK/Kernel_OpenCL.h>
#endif
