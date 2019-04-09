
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/Event_CUDA.cpp

#define __CLASS__ "Event_CUDA::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== C ==================================================================
#include <stdio.h>
#include <time.h>

// ===== OpenNet/CUDA =======================================================
#include "CUW.h"
#include "Event_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

Event_CUDA::Event_CUDA() : mEnd( NULL ), mStart( NULL )
{
    // printf( __CLASS__ "Event_CUDA()\n" );
}

// aProfiling  false  Create the CUevent without enabling profiling
//             true   Create the CUevent with profiling enabled
//
// Exception  KmsLib::Exception *  See Init
Event_CUDA::Event_CUDA( bool aProfiling ) : Event( aProfiling ), mEnd( NULL ), mStart( NULL )
{
    Init( aProfiling );
}

Event_CUDA::~Event_CUDA()
{
    if ( NULL != mEnd )
    {
        CUW_EventDestroy( mEnd );

        if ( NULL != mStart )
        {
            CUW_EventDestroy( mStart );
        }
    }
}

// aStream  The stream used to execute the kernel

// CRITICAL PATH  Processing
//                1 / iteration
void Event_CUDA::RecordEnd( CUstream aStream )
{
    assert( NULL != aStream );

    assert( NULL != mEnd );

    CUW_EventRecord( mEnd, aStream );

    if ( mProfilling )
    {
        struct timespec lNow;

        int lRet = clock_gettime( CLOCK_MONOTONIC, & lNow );
        assert( 0 == lRet );

        mQueuedTime = ( lNow.tv_sec * 1000000 ) + ( lNow.tv_nsec / 1000 );
    }
}

// aStream  The stream used to execute the kernel

// CRITICAL PATH  Processing.Profiling
//                1 / iteration
void Event_CUDA::RecordStart( CUstream aStream )
{
    assert( NULL != aStream );

    assert( NULL != mStart );

    CUW_EventRecord( mStart, aStream );
}

// ===== Event ==============================================================

void Event_CUDA::Init( bool aProfiling )
{
    Event::Init( aProfiling );

    if ( mProfilling )
    {
        CUW_EventCreate( & mEnd  , CU_EVENT_BLOCKING_SYNC );
        CUW_EventCreate( & mStart, CU_EVENT_DEFAULT       );
    }
    else
    {
        CUW_EventCreate( & mEnd  , CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING );
    }

}

void Event_CUDA::Wait()
{
    assert( NULL != mEnd   );

    CUW_EventSynchronize( mEnd );

    if ( mProfilling )
    {
        assert(    0  < mQueuedTime );
        assert( NULL != mStart      );

        struct timespec lNow;

        int lRet = clock_gettime( CLOCK_MONOTONIC, & lNow );
        assert( 0 == lRet );

        uint64_t lCompletedTime = ( lNow.tv_sec * 1000000 ) + ( lNow.tv_nsec / 1000 );
        uint64_t lTotal_us = lCompletedTime - mQueuedTime;

        float lExecution_ms;
    
        CUW_EventElapsedTime( & lExecution_ms, mStart, mEnd );

        mExecution_us = lExecution_ms * 1000.0;

        if ( lTotal_us > mExecution_us )
        {
            mQueued_us = lTotal_us - mExecution_us;
        }
        else
        {
            mQueued_us = 0;
        }
    }
}
