
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/UserBuffer_CUDA.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>

// ===== OpenNet ============================================================
#include "CUW.h"

#include "UserBuffer_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aSize_byte
//
// Exception  KmsLib::Exception 
UserBuffer_CUDA::UserBuffer_CUDA(unsigned int aSize_byte) : UserBuffer_Internal(aSize_byte)
{
    assert( 0 < aSize_byte );

    CUW_MemAlloc( & mMem_DA, aSize_byte );

    Clear_Internal();
}

// ===== UserBuffer_Internal ================================================

void UserBuffer_CUDA::Clear_Internal()
{
    assert( 0 != mMem_DA    );
    assert( 0 <  mSize_byte );

    CUW_MemsetD8( mMem_DA, 0, mSize_byte );
}

void UserBuffer_CUDA::Read_Internal(unsigned int aOffset_byte, void * aOut, unsigned int aSize_byte)
{
    assert(NULL != aOut      );
    assert(   0 <  aSize_byte);

    assert( 0 != mMem_DA );

    CUW_MemcpyDtoH( aOut, mMem_DA + aOffset_byte, aSize_byte );
}

void UserBuffer_CUDA::Write_Internal(unsigned int aOffset_byte, const void * aIn, unsigned int aSize_byte)
{
    assert(NULL != aIn       );
    assert(0    <  aSize_byte);

    assert( 0 != mMem_DA );

    CUW_MemcpyHtoD( mMem_DA + aOffset_byte, aIn, aSize_byte );
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet::UserBuffer ================================================

UserBuffer_CUDA::~UserBuffer_CUDA()
{
    assert( 0 != mMem_DA );

    CUW_MemFree( mMem_DA );
}
