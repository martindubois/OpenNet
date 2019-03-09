
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/UserBuffer_OpenCL.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>

// ===== OpenNet ============================================================
#include "OCLW.h"

#include "UserBuffer_OpenCL.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aSize_byte
// aContext [---;RW-]
//
// Exception  KmsLib::Exception *
UserBuffer_OpenCL::UserBuffer_OpenCL(unsigned int aSize_byte, cl_context aContext, cl_command_queue aCommandQueue) : UserBuffer_Internal(aSize_byte), mCommandQueue(aCommandQueue)
{
    assert(   0 < aSize_byte    );
    assert(NULL != aContext     );
    assert(NULL != aCommandQueue);

    mMem = OCLW_CreateBuffer(aContext, CL_MEM_READ_WRITE, aSize_byte);
    assert(NULL != mMem);

    Clear_Internal();
}

// ===== UserBuffer_Internal ================================================

void UserBuffer_OpenCL::Clear_Internal()
{
    assert(0 < mSize_byte);

    void * lTemp = new unsigned char[mSize_byte];
    assert(NULL != lTemp);

    memset(lTemp, 0, mSize_byte);

    try
    {
        OCLW_EnqueueWriteBuffer(mCommandQueue, mMem, CL_TRUE, 0, mSize_byte, lTemp, 0, NULL, NULL);
    }
    catch (...)
    {
        delete[] lTemp;
        throw;
    }

    delete[] lTemp;
}

void UserBuffer_OpenCL::Read_Internal(unsigned int aOffset_byte, void * aOut, unsigned int aSize_byte)
{
    assert(NULL != aOut      );
    assert(   0 <  aSize_byte);

    assert(NULL != mCommandQueue);

    OCLW_EnqueueReadBuffer(mCommandQueue, mMem, true, aOffset_byte, aSize_byte, aOut, 0, NULL, NULL);
}

void UserBuffer_OpenCL::Write_Internal(unsigned int aOffset_byte, const void * aIn, unsigned int aSize_byte)
{
    assert(NULL != aIn       );
    assert(0    <  aSize_byte);

    assert(NULL != mCommandQueue);

    OCLW_EnqueueWriteBuffer(mCommandQueue, mMem, true, aOffset_byte, aSize_byte, aIn, 0, NULL, NULL);
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet::UserBuffer ================================================

UserBuffer_OpenCL::~UserBuffer_OpenCL()
{
    assert(NULL != mCommandQueue);
    assert(NULL != mMem         );

    OCLW_ReleaseMemObject   (mMem         );
    OCLW_ReleaseCommandQueue(mCommandQueue);
}
