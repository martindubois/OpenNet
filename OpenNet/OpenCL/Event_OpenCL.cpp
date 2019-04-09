
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/OpenCL/Event_OpenCL.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== C ==================================================================
#include <stdint.h>

// ===== OpenNet/OpenCL =====================================================
#include "OCLW.h"

#include "Event_OpenCL.h"

// Static functions declaration
/////////////////////////////////////////////////////////////////////////////

static uint64_t GetEventProfilingInfo(cl_event aEvent, cl_profiling_info aParam);

// Public
/////////////////////////////////////////////////////////////////////////////

Event_OpenCL::Event_OpenCL()
{
}

// aProfiling  Stored in the base class and used to know if profiling
//             information must be retrieved after waiting.
Event_OpenCL::Event_OpenCL( bool aProfiling ) : Event( aProfiling )
{
}

// ===== Event ==============================================================

void Event_OpenCL::Wait()
{
    assert( NULL != mEvent );

    OCLW_WaitForEvents( 1, & mEvent );

    if ( mProfilling )
    {
        uint64_t lQueued    = GetEventProfilingInfo( mEvent, CL_PROFILING_COMMAND_QUEUED );
        uint64_t lSubmitted = GetEventProfilingInfo( mEvent, CL_PROFILING_COMMAND_SUBMIT );
        uint64_t lStart     = GetEventProfilingInfo( mEvent, CL_PROFILING_COMMAND_START  );
        uint64_t lEnd       = GetEventProfilingInfo( mEvent, CL_PROFILING_COMMAND_END    );

        mExecution_us = ( lEnd       - lStart     ) / 1000;
        mQueued_us    = ( lSubmitted - lQueued    ) / 1000;
        mSubmitted_us = ( lStart     - lSubmitted ) / 1000;
    }

    OCLW_ReleaseEvent( mEvent );
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aEvent [---;R--]
// aParam
//
// Return  This method return the retrieved information
//
// Exception  KmsLib::Exception *  See OCLW_GetEventProfilingInfo
// Thread     Worker

// CRITICAL PATH  Processing.Profiling
//                4 / iteration
uint64_t GetEventProfilingInfo(cl_event aEvent, cl_profiling_info aParam)
{
    assert(NULL != aEvent);

    uint64_t lResult;

    OCLW_GetEventProfilingInfo(aEvent, aParam, sizeof(lResult), &lResult);

    return lResult;
}
