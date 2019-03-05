
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Event_OpenCL.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "Event_OpenCL.h"

// Public
/////////////////////////////////////////////////////////////////////////////

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
