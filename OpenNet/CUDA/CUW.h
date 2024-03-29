
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUW.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== NVIDIA =============================================================
#include <cuda.h>

// Functions
/////////////////////////////////////////////////////////////////////////////

extern void CUW_Check();

extern void CUW_DeviceGetAttribute ( int * aValue, CUdevice_attribute aAttribute, CUdevice aDevice );
extern void CUW_DeviceGetCount     ( int * aCount );
extern void CUW_DeviceGetName      ( char * aName, int aSize_byte, CUdevice aDevice );
extern void CUW_DeviceTotalMem     ( size_t * aSize_byte, CUdevice aDevice );
extern void CUW_EventElapsedTime   ( float * aElapsed_ms, CUevent aStart, CUevent aEnd );
extern void CUW_FuncGetAttribute   ( int * aOut, CUfunction_attribute aAttr, CUfunction aFunction );
extern void CUW_Init               ( unsigned int aFlags );
extern void CUW_MemcpyDtoH         ( void * aDst, CUdeviceptr aSrc_DA, size_t aSize_byte );
extern void CUW_PointerSetAttribute( const void * aValue, CUpointer_attribute aAttribute, CUdeviceptr aPtr_DA );

// ===== CUcontext ==========================================================
extern void CUW_CtxCreate             ( CUcontext * aContext, unsigned int aFlags, CUdevice aDevice );
extern void CUW_CtxDestroy            ( CUcontext   aContext );
extern void CUW_CtxPopCurrent         ( CUcontext * aContext );
extern void CUW_CtxPushCurrent        ( CUcontext   aContext );
extern void CUW_CtxSetCurrent         ( CUcontext   aContext );
extern void CUW_DevicePrimaryCtxRetain( CUcontext * aContext, CUdevice aDevice );

// ===== CUdevice ===========================================================
extern void CUW_DeviceGet              ( CUdevice * aDevice, int aIndex );
extern void CUW_DevicePrimaryCtxRelease( CUdevice   aDevice );

// ===== CUdeviceptr ========================================================
extern void CUW_MemAlloc  ( CUdeviceptr * aPtr_DA, size_t aSize_byte );
extern void CUW_MemcpyHtoD( CUdeviceptr   aPtr_DA, const void * aSrc, size_t aSize_byte );
extern void CUW_MemFree   ( CUdeviceptr   aPtr_DA );
extern void CUW_MemsetD8  ( CUdeviceptr   aPtr_DA, unsigned char aValue, size_t aSize_byte );

// ===== CUevent ============================================================
extern void CUW_EventCreate     ( CUevent * aEvent, unsigned int aFlags );
extern void CUW_EventDestroy    ( CUevent   aEvent );
extern void CUW_EventRecord     ( CUevent   aEvent, CUstream aStream );
extern void CUW_EventSynchronize( CUevent   aEvent );

// ===== CUfunction =========================================================
extern void CUW_LaunchKernel     ( CUfunction aFunction, unsigned int aGridDimX, unsigned int aGridDimY, unsigned int aGridDimZ, unsigned int aBlockDimX, unsigned int aBlockDimY, unsigned int aBlockDimZ, unsigned int aSharedMemSize_byte, CUstream aStream, void * * aArguments, void * * aExtra );
extern void CUW_ModuleGetFunction( CUfunction * aFunction, CUmodule aModule, const char * aName );

// ===== CUmodule ===========================================================
extern void CUW_ModuleLoadDataEx( CUmodule * aModule, const void * aImage, unsigned int aNumOptions, CUjit_option * aOptions, void * * aOptionValues );
extern void CUW_ModuleUnload    ( CUmodule   aModule );

// ===== CUstream ===========================================================
extern void CUW_LaunchHostFunc( CUstream   aStream, CUhostFn aFunction, void * aUserData );
extern void CUW_StreamCreate  ( CUstream * aStream, unsigned int aFlags );
extern void CUW_StreamDestroy ( CUstream   aStream );
