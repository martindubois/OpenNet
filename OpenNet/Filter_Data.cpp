
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Filter_Data.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== OpenNet ============================================================
#include "OCLW.h"

#include "Filter_Data.h"

// Public
/////////////////////////////////////////////////////////////////////////////

Filter_Data::Filter_Data() : mCommandQueue(NULL), mFilter(NULL), mKernel(NULL), mProgram(NULL)
{
}

// Exception  KmsLib::Exception *  See OCLW_ReleaseCommandQueue
//                                 See OCLW_ReleaseKernel
//                                 See OCLW_ReleaseProgram
void Filter_Data::Release()
{
    assert(NULL != mCommandQueue);
    assert(NULL != mKernel      );
    assert(NULL != mProgram     );

    // OCLW_CreateCommandQueueWithProperties ==> OCLW_ReleaseCommandQueue  See Processing_Create
    OCLW_ReleaseCommandQueue(mCommandQueue);
    mCommandQueue = NULL;

    // OCLW_CreateKernel ==> OCLW_ReleaseKernel  See Processing_Create
    OCLW_ReleaseKernel(mKernel);
    mKernel = NULL;

    // OCLW_CreateProgramWithSournce ==> OCLW_ReleaseProgram  See Process_Create
    OCLW_ReleaseProgram(mProgram);
    mProgram = NULL;
}

void Filter_Data::Reset()
{
    assert(NULL == mCommandQueue);
    assert(NULL == mKernel      );
    assert(NULL == mProgram     );

    mCommandQueue = NULL;
    mFilter       = NULL;
    mKernel       = NULL;
    mProgram      = NULL;
}
