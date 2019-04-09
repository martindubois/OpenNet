
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/OpenCL/Event_OpenCL.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== OpenNet/OpenCL =====================================================
#include "../Event.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Event_OpenCL : public Event
{

public:

    Event_OpenCL();

    Event_OpenCL( bool aProfiling );

    // ===== Event ==========================================================
    virtual void Wait();

    cl_event mEvent;

};
