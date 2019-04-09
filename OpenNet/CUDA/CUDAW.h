
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDAW.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== NVIDIA =============================================================
#include <cuda_runtime.h>

// Functions
/////////////////////////////////////////////////////////////////////////////

extern void CUDAW_Free               ( void * aMemory_DA );
extern void CUDAW_GetDeviceCount     ( int * aCount );
extern void CUDAW_GetDeviceProperties( cudaDeviceProp * aProp, int aDevice );
extern void CUDAW_LaunchKernel       ( const void * aFunc, dim3 aGridDim, dim3 aBlockDim, void ** aArgs, size_t aSharedMem_byte, cudaStream_t aStream );
extern void CUDAW_Malloc             ( void * * aMemory_DA, size_t aSize_byte );
extern void CUDAW_SetDevice          ( int aDevice );
extern void CUDAW_StreamCreate       ( cudaStream_t * aStream );
extern void CUDAW_StreamDestroy      ( cudaStream_t   aStream );