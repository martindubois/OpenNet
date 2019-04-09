
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/Thread_CUDA.cpp

#define __CLASS__ "Thread_CUDA::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"
#include <KmsBase.h>

// ===== C ==================================================================
#include <stdint.h>

// ===== OpenNet/CUDA =======================================================

#include "../Linux/Adapter_Linux.h"

#include "CUW.h"
#include "Event_CUDA.h"

#include "Thread_CUDA.h"

// Protected
/////////////////////////////////////////////////////////////////////////////

// aProcessor [-K-;RW-]
// aModule    [-KO;RW-]
Thread_CUDA::Thread_CUDA( Processor_Internal * aProcessor, CUmodule aModule ) : mArguments( NULL ), mFunction( NULL ), mModule( aModule ), mProcessor_CUDA( dynamic_cast< Processor_CUDA * >( aProcessor ) ), mStream( NULL )
{
    assert( NULL != aProcessor );

    assert( NULL != mProcessor_CUDA );
}

Thread_CUDA::~Thread_CUDA()
{
}

// aAdapters [---;RW-]
// aBuffers  [---;RW-]
// aProfiling
//
// Exception  KmsLib::Exception *  See CUW_StreamCreate, CUW_ModuleFunction
//                                 and Adapter_Linux::Buffers_Allocate
//
// Thread_CUDA::Prepare ==> Thread_CUDA::Release
void Thread_CUDA::Prepare( Adapter_Vector * aAdapters, Buffer_Internal_Vector * aBuffers, bool aProfiling )
{
    assert( NULL != aAdapters         );
    assert(    0 <  aAdapters->size() );
    assert( NULL != aBuffers          );

    assert( NULL != mModule         );
    assert( NULL != mProcessor_CUDA );
    assert( NULL == mStream         );

    mProcessor_CUDA->SetContext();

    // CUDA_StreamCreate ==> CUDA_StreamDestroy  See Release
    CUW_StreamCreate( & mStream, CU_STREAM_NON_BLOCKING );
    assert( NULL != mStream );

    CUW_ModuleGetFunction( & mFunction, mModule, "Filter" );
    assert( NULL != mFunction );

    unsigned int i;

    for ( i = 0; i < aAdapters->size(); i++)
    {
        assert(NULL != (*aAdapters)[i]);

        Adapter_Linux * lAdapter = dynamic_cast<Adapter_Linux *>((*aAdapters)[i]);
        assert(NULL != lAdapter);

        lAdapter->Buffers_Allocate( aProfiling, aBuffers );
    }

    assert( 0 < aBuffers->size() );
}

// aKernel     [---;RW-]
// aEvent      [---;RW-]
// aGlobalSize [---;R--]
// aLocalSize  [--O;R--]
// aArguments  [---;R--]
//
// Exception  KmsLib::Exception *  See CUW_LaunchKernel and
//                                 CUW_LaunchHostFunction
// Thread     Worker
//
// Processing_Queue ==> Event_CUDA::Wait

// CRITICAL PATH  Processing
//                1 / iteration
void Thread_CUDA::Processing_Queue( OpenNet::Kernel * aKernel, Event * aEvent, const size_t * aGlobalSize, const size_t * aLocalSize, void * * aArguments )
{
    assert( NULL != aKernel     );
    assert( NULL != aEvent      );
    assert( NULL != aGlobalSize );
    assert( NULL != aArguments  );

    assert( NULL != mFunction );
    assert( NULL != mStream   );

    Event_CUDA * lEvent = dynamic_cast< Event_CUDA * >( aEvent );
    assert( NULL != lEvent );

    if ( aKernel->IsProfilingEnabled() )
    {
        lEvent->RecordStart( mStream );
    }

    CUW_LaunchKernel( mFunction, aGlobalSize[ 0 ], 1, 1, ( NULL == aLocalSize ) ? aGlobalSize[ 0 ] : aLocalSize[ 0 ], 1, 1, 0, mStream, aArguments, NULL );

    lEvent->RecordEnd( mStream );
}

// aKernel    [---;RW-]
//
// Exception  KmsLib::Exception *  See CUW_StreamDestroy
//
// Thread_CUDA::Prepare ==> Thread_CUDA::Release
void Thread_CUDA::Release( OpenNet::Kernel * aKernel)
{
    assert(NULL != aKernel);

    assert( NULL != mProcessor_CUDA );

    if ( NULL != mStream )
    {
        mProcessor_CUDA->SetContext();

        if ( NULL != mArguments )
        {
            // printf( __CLASS__ "Release - delete [] 0x%lx (mArguments)\n", reinterpret_cast< uint64_t >( mArguments ) );

            // new ==> delete  See Thread_Function_CUDA::Prepare or
            //                 Thread_Kernel_CUDA::Prepare
            delete [] mArguments;
        }

        // CUW_StreamCreate ==> CUW_StreamDestroy  See Prepare_Internal
        CUW_StreamDestroy( mStream );
    }
}

// Thread  Worker
void Thread_CUDA::Run_Start()
{
    assert( NULL != mProcessor_CUDA );

    mProcessor_CUDA->SetContext();
}
