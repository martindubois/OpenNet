
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Thread.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

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
    void SetProgram(cl_program        aProgram);

    virtual void Delete();

    virtual void Prepare();

    void Stop_Wait   (TryToSolveHang aTryToSolveHang, void * aContext);

protected:

    virtual ~Thread();

    void Processing_Queue(const size_t * aGlobalSize, const size_t * aLocalSize, cl_event * aEvent);
    void Processing_Wait (cl_event aEvent);

    virtual void Processing_Queue(unsigned int aIndex) = 0;
    virtual void Processing_Wait (unsigned int aIndex) = 0;

    virtual void Release();

    void Run_Iteration(unsigned int aIndex);

    virtual void Run_Loop () = 0;
    virtual void Run_Start() = 0;

    // ===== KmsLib::ThreadBase =============================================

    virtual unsigned int Run();

    Adapter_Vector       mAdapters    ;
    Buffer_Data_Vector   mBuffers     ;
    cl_command_queue     mCommandQueue;
    KmsLib::DebugLog   * mDebugLog    ;
    OpenNet::Kernel    * mKernel      ;
    cl_kernel            mKernel_CL   ;
    Processor_Internal * mProcessor   ;

private:

    void Run_Wait();

    cl_program        mProgram ;

};
