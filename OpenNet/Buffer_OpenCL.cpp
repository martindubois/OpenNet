
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Buffer_OpenCL.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdint.h>

// ===== OpenNet ============================================================
#include "OCLW.h"

#include "Buffer_OpenCL.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProfiling              Set to true when profiling data must be captured
// aMem          [DK-;RW-] The cl_mem instance describing the buffer
// aCommandQueue [-K-;RW-] The command queue to use for Read and Write
//                         operation
// aPacketQty              The number of packet the buffer contains
Buffer_OpenCL::Buffer_OpenCL( bool aProfiling, cl_mem aMem, cl_command_queue aCommandQueue, unsigned int aPacketQty)
    : Buffer_Internal( aPacketQty, static_cast< Event * >( & mEvent_OpenCL ) )
    , mCommandQueue( aCommandQueue )
    , mEvent       ( NULL          )
    , mEvent_OpenCL( aProfiling    )
    , mMem         ( aMem          )
{
    assert(NULL != aMem      );
    assert(   0 <  aPacketQty);
}

// ===== Buffer_Internal ====================================================

Buffer_OpenCL::~Buffer_OpenCL()
{
    assert(NULL != mMem);

    if (NULL != mEvent)
    {
        OCLW_ReleaseEvent(mEvent);
    }

    // OCLW_CreateBuffer ==> OCLW_ReleaseMemObject  See ?
    OCLW_ReleaseMemObject(mMem);
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Buffer_Internal ====================================================

void Buffer_OpenCL::Read(unsigned int aOffset_byte, void * aOut, unsigned int aSize_byte)
{
    assert(NULL != aOut      );
    assert(   0 <  aSize_byte);

    assert(NULL != mCommandQueue);
    assert(NULL != mMem         );

    if (NULL != mEvent)
    {
        ReleaseEvent();
    }

    OCLW_EnqueueReadBuffer(mCommandQueue, mMem, CL_FALSE, aOffset_byte, aSize_byte, aOut, 0, NULL, &mEvent);
    assert(NULL != mEvent);

    OCLW_Flush(mCommandQueue);
}

void Buffer_OpenCL::Write(unsigned int aOffset_byte, const void * aIn, unsigned int aSize_byte)
{
    assert(NULL != aIn       );
    assert(0    <  aSize_byte);

    assert(NULL != mCommandQueue);
    assert(NULL != mMem         );

    if (NULL != mEvent)
    {
        ReleaseEvent();
    }

    OCLW_EnqueueWriteBuffer(mCommandQueue, mMem, CL_FALSE, aOffset_byte, aSize_byte, aIn, 0, NULL, &mEvent);
    assert(NULL != mEvent);

    OCLW_Flush(mCommandQueue);
}

void Buffer_OpenCL::Wait_Internal()
{
    if (NULL != mEvent)
    {
        OCLW_WaitForEvents(1, &mEvent);

        ReleaseEvent();
    }
}

// Private
/////////////////////////////////////////////////////////////////////////////

inline void Buffer_OpenCL::ReleaseEvent()
{
    OCLW_ReleaseEvent(mEvent);
    mEvent = NULL;
}
