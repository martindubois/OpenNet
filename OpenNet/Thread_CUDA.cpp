
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread_CUDA.cpp

#define __CLASS__ "Thread_CUDA::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== OpenNet ============================================================
#include "Adapter_Linux.h"
#include "CUW.h"

#include "Thread_CUDA.h"

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
static void CUDA_CB KernelCompleted( void * aUserData );

// Internal
/////////////////////////////////////////////////////////////////////////////

void Thread_CUDA::KernelCompleted()
{
    int lRet = sem_post( & mSemaphore );
    assert( 0 == lRet );
}

// Protected
/////////////////////////////////////////////////////////////////////////////

Thread_CUDA::Thread_CUDA() : mArguments( NULL ), mFunction( NULL ), mModule( NULL ), mStream( NULL )
{
}

Thread_CUDA::Thread_CUDA( CUmodule aModule ) : mArguments( NULL ), mFunction( NULL ), mModule( aModule ), mStream( NULL )
{
    assert( NULL != aModule );
}

Thread_CUDA::~Thread_CUDA()
{
}

// aAdapters  [---;RW-]
// aBuffers   [---;RW-]
//
// Exception  KmsLib::Exception *  See CUW_StreamCreate, CUW_ModuleFunction
//                                 and Adapter_Linux::Buffers_Allocate
//
// Thread_CUDA::Prepare ==> Thread_CUDA::Release
void Thread_CUDA::Prepare( Adapter_Vector * aAdapters, Buffer_Data_Vector * aBuffers )
{
    assert( NULL != aAdapters );
    assert( NULL != aBuffers  );

    Prepare_Internal( aAdapters, aBuffers );

    assert( 0 < aBuffers->size() );

    // sem_init ==> sem_destroy  See Release
    int lRet = sem_init( & mSemaphore, 0, aBuffers->size() );
    assert( 0 == lRet );
}

// aAdapters  [---;RW-]
// aBuffers   [---;RW-]
// aQueueDepth
//
// Exception  KmsLib::Exception *  See CUW_StreamCreate, CUW_ModuleFunction
//                                 and Adapter_Linux::Buffers_Allocate
//
// Thread_CUDA::Prepare ==> Thread_CUDA::Release
void Thread_CUDA::Prepare( Adapter_Vector * aAdapters, Buffer_Data_Vector * aBuffers, unsigned int aQueueDepth )
{
    assert( NULL != aAdapters   );
    assert( NULL != aBuffers    );
    assert(    0 <  aQueueDepth );

    Prepare_Internal( aAdapters, aBuffers );

    // sem_init ==> sem_destroy  See Release
    int lRet = sem_init( & mSemaphore, 0, aQueueDepth );
    assert( 0 == lRet );
}

// aKernel     [---;RW-]
// aGlobalSize [---;R--]
// aLocalSize  [--O;R--]
// aArguments  [---;R--]
//
// Exception      KmsLib::Exception *  See CUW_LaunchKernel and
//                                     CUW_LaunchHostFunction
// CRITICAL PATH  Buffer-
//
// Processing_Queue ==> Processing_Wait
void Thread_CUDA::Processing_Queue( OpenNet::Kernel * aKernel, const size_t * aGlobalSize, const size_t * aLocalSize, void * * aArguments )
{
    // printf( __CLASS__ "Processing_Queue( , , ,  )\n" );

    assert( NULL != aKernel     );
    assert( NULL != aGlobalSize );
    assert( NULL != aArguments  );

    assert( NULL != mFunction );
    assert( NULL != mStream   );

    usleep( 10000 );

    CUW_LaunchKernel( mFunction, aGlobalSize[ 0 ], 1, 1, ( NULL == aLocalSize ) ? aGlobalSize[ 0 ] : aLocalSize[ 0 ], 1, 1, 0, mStream, aArguments, NULL );
    CUW_LaunchHostFunction( mStream, ::KernelCompleted, this );
}

// Exception      KmsLib::Exception *  CODE_SYSTEM_ERROR
// CRITICAL PATH  Buffer-
//
// Processing_Queue ==> Processing_Wait
void Thread_CUDA::Processing_Wait()
{
    int lRet = sem_wait( & mSemaphore );
    if ( 0 != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_SYSTEM_ERROR,
            "sem_wait(  ) failed", NULL, __FILE__, __CLASS__ "Processing_Wait", __LINE__, lRet );
    }
}

// aKernel [---;RW-]
//
// Exception  KmsLib::Exception *  See CUW_StreamDestroy
//
// Thread_CUDA::Prepare ==> Thread_CUDA::Release
void Thread_CUDA::Release(OpenNet::Kernel * aKernel)
{
    assert(NULL != aKernel);

    if ( NULL != mStream )
    {
        if ( NULL != mArguments )
        {
            // sem_init ==> sem_destroy  See Prepare
            int lRet = sem_destroy( & mSemaphore );
            assert( 0 == lRet );

            // printf( __CLASS__ "Release - delete [] 0x%lx (mArguments)\n", reinterpret_cast< uint64_t >( mArguments ) );

            // new ==> delete  See Thread_Function_CUDA::Prepare or
            //                 Thread_Kernel_CUDA::Prepare
            delete [] mArguments;
        }

        // CUW_StreamCreate ==> CUW_StreamDestroy  See Prepare_Internal
        CUW_StreamDestroy( mStream );
    }
}

// aProcessor [---;RW-]
//
// Thread  Worker
void Thread_CUDA::Run_Start( Processor_Internal * aProcessor )
{
    assert( NULL != aProcessor );

    Processor_CUDA * lProcessor = dynamic_cast< Processor_CUDA * >( aProcessor );
    assert( NULL != lProcessor );

    lProcessor->SetContext();
}

// Private
/////////////////////////////////////////////////////////////////////////////

void Thread_CUDA::Prepare_Internal( Adapter_Vector * aAdapters, Buffer_Data_Vector * aBuffers )
{
    assert( NULL != aAdapters         );
    assert(    0 <  aAdapters->size() );
    assert( NULL != aBuffers          );

    assert( NULL != mModule );
    assert( NULL == mStream );

    // CUDA_StreamCreate ==> CUDA_StreamDestroy  See Release
    CUW_StreamCreate( & mStream, CU_STREAM_NON_BLOCKING );
    assert( NULL != mStream );

    CUW_ModuleGetFunction( & mFunction, mModule, "Filter" );
    assert( NULL != mFunction );

    for (unsigned int i = 0; i < aAdapters->size(); i++)
    {
        assert(NULL != (*aAdapters)[i]);

        Adapter_Linux * lAdapter = dynamic_cast<Adapter_Linux *>((*aAdapters)[i]);
        assert(NULL != lAdapter);

        lAdapter->Buffers_Allocate( aBuffers );
    }
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================

void CUDA_CB KernelCompleted( void * aUserData )
{
    assert( NULL != aUserData );

    Thread_CUDA * lThis = reinterpret_cast< Thread_CUDA * >( aUserData );

    lThis->KernelCompleted();
}
