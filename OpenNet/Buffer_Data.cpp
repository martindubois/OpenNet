
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Buffer_Data.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== OpenNet ============================================================
#include "OCLW.h"

#include "Buffer_Data.h"

// Public
/////////////////////////////////////////////////////////////////////////////

Buffer_Data::Buffer_Data() : mEvent(NULL), mMem(NULL), mPacketQty(0), mMarkerValue(0)
{
}

// Exception  KmsLib::Exception *  See OCLW_GetEventProfilingInfo
uint64_t Buffer_Data::GetEventProfilingInfo(cl_profiling_info aParam)
{
    assert(NULL != mEvent);

    uint64_t lResult;

    OCLW_GetEventProfilingInfo(mEvent, aParam, sizeof(lResult), &lResult);

    return lResult;
}

unsigned int Buffer_Data::GetMarkerValue()
{
    mMarkerValue++;

    return mMarkerValue;
}

// Exception  KmsLib::Exception *  See OCLW_ReleaseEvent
void Buffer_Data::ReleaseEvent()
{
    assert(NULL != mEvent);

    // OCLW_EnqueueNDRangeKernel ==> OCLW_ReleaseEvent  See ?
    OCLW_ReleaseEvent(mEvent);

    mEvent = NULL;
}

// Exception  KmsLib::Exception *  See OCLW_ReleaseMemObject
void Buffer_Data::ReleaseMemObject()
{
    assert(NULL != mMem);

    // OCLW_CreateBuffer ==> OCLW_ReleaseMemObject  See ?
    OCLW_ReleaseMemObject(mMem);
    mMem = NULL;
}

void Buffer_Data::Reset()
{
    assert(NULL == mEvent);
    assert(NULL == mMem  );

    mEvent       = NULL;
    mMarkerValue =    0;
    mMem         = NULL;
    mPacketQty   =    0;
}

void Buffer_Data::ResetMarkerValue()
{
    mMarkerValue = 0;
}

// Exception  KmsLib::Exception *  See OCLW_WaitForEvents
void Buffer_Data::WaitForEvent()
{
    assert(NULL != mEvent);

    OCLW_WaitForEvents(1, &mEvent);
}
