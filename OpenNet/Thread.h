
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Thread.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== Import/Includes ====================================================
#include <KmsLib/ThreadBase.h>

// ===== OpenNet ============================================================
#include "Internal/Adapter_Internal.h"
#include "Internal/Processor_Internal.h"

class Buffer_Internal;

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread : public KmsLib::ThreadBase
{

public:

    typedef void(*TryToSolveHang)(void *, Adapter_Internal *);

    Thread(Processor_Internal * aProcessor, KmsLib::DebugLog * aDebugLog);

    void AddAdapter(Adapter_Internal * aAdapter);

    Buffer_Internal * GetBuffer(Adapter_Internal * aAdapter, unsigned int aIndex);

    OpenNet::Kernel * GetKernel();

    OpenNet::Status GetStatistics(unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);

    OpenNet::Status ResetStatistics();

    void SetKernel (OpenNet::Kernel * aKernel );

    virtual void Delete();

    virtual void Prepare();

    void Stop_Wait   (TryToSolveHang aTryToSolveHang, void * aContext);

protected:

    virtual ~Thread();

    void Processing_Wait( OpenNet::Kernel * aKernel, Event * aEvent );

    // aIndex
    //
    // Thread  Worker
    //
    // Processing_Queue ==> Processing_Wait

    // CRITICAL PATH  Processing
    //                1 / iteration
    virtual void Processing_Queue(unsigned int aIndex) = 0;

    // aIndex
    //
    // Thread  Worker
    //
    // Processing_Queue ==> Processing_Wait

    // CRITICAL PATH  Processing
    //                1 / iteration
    virtual void Processing_Wait(unsigned int aIndex) = 0;

    // Threads  Apps
    virtual void Release() = 0;

    virtual void Run_Loop ();
    virtual void Run_Start();
    virtual void Run_Wait ();

    // ===== KmsLib::ThreadBase =============================================

    virtual unsigned int Run();

    Adapter_Vector         mAdapters  ;
    Buffer_Internal_Vector mBuffers   ;
    KmsLib::DebugLog     * mDebugLog  ;
    OpenNet::Kernel      * mKernel    ;
    Processor_Internal   * mProcessor ;
    unsigned int           mQueueDepth;

};
