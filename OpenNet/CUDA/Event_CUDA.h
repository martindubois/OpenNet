
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/CUDA/Event_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== NVIDIA =============================================================
#include <cuda.h>

// ===== OpenNet/CUDA =======================================================
#include "../Event.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Event_CUDA : public Event
{

public:

    Event_CUDA();

    Event_CUDA( bool aProfiling );

    ~Event_CUDA();

    void RecordStart( CUstream aStream );
    void RecordEnd  ( CUstream aStream );

    // ===== Event ==========================================================

    virtual void Init( bool aProfiling );

    virtual void Wait();

private:

    uint64_t mQueuedTime;
    CUevent  mStart     ;
    CUevent  mEnd       ;

};
