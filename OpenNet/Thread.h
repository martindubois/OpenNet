
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
#include "Adapter_Internal.h"
#include "Processor_Internal.h"

class Buffer_Data;

// Class
/////////////////////////////////////////////////////////////////////////////

class Thread : public KmsLib::ThreadBase
{

public:

    typedef void(*TryToSolveHang)(void *, Adapter_Internal *);

    Thread(Processor_Internal * aProcessor, KmsLib::DebugLog * aDebugLog);

    void AddAdapter(Adapter_Internal * aAdapter);

    OpenNet::Kernel * GetKernel();

    OpenNet::Status GetStatistics(unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);

    OpenNet::Status ResetStatistics();

    void SetKernel (OpenNet::Kernel * aKernel );

    virtual void Delete();

    virtual void Prepare();

    void Stop_Wait   (TryToSolveHang aTryToSolveHang, void * aContext);

protected:

    virtual ~Thread();

    virtual void Processing_Queue(unsigned int aIndex) = 0;
    virtual void Processing_Wait (unsigned int aIndex) = 0;

    void Run_Iteration(unsigned int aIndex);

    virtual void Release() = 0;

    virtual void Run_Loop () = 0;
    virtual void Run_Start() = 0;

    // ===== KmsLib::ThreadBase =============================================

    virtual unsigned int Run();

    Adapter_Vector       mAdapters    ;
    Buffer_Data_Vector   mBuffers     ;
    KmsLib::DebugLog   * mDebugLog    ;
    OpenNet::Kernel    * mKernel      ;
    Processor_Internal * mProcessor   ;

private:

    void Run_Wait();

};
