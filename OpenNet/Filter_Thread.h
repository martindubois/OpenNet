
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/FilterThread.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "Adapter_Internal.h"
#include "Filter_Data.h"
#include "Processor_Internal.h"

class Buffer_Data;

// Class
/////////////////////////////////////////////////////////////////////////////

class FilterThread
{

public:

    typedef void(*TryToSolveHang)(void *, Adapter_Internal *);

    FilterThread(TryToSolveHang aTryToSolveHang, void * aContext);

    ~FilterThread();

    void AddAdapter(Adapter_Internal aAdapter);

    void SetProcessor(Processor_Internal aProcessor);

    void Start       ();
    void Stop_Request();
    void Stop_Wait   ();

// internal:

    virtual void Run();

private:

    typedef std::vector<Buffer_Data *> Buffer_Vector;

    void State_Change(unsigned int aFrom, unsigned int aTo);

    void Stop_Request_Zone0();
    void Stop_Wait_Zone0   ();

    Adapter_Vector       mAdapters  ;
    Buffer_Vector        mBuffers   ;
    Filter_Data          mFilterData;
    Processor_Internal * mProcessor ;
    HANDLE               mThread    ;
    DWORD                mThreadId  ;

    // ===== Zone 0 =========================================================
    // Threads  Apps
    //          Worker
    CRITICAL_SECTION mZone0;

    unsigned int mState;

};
