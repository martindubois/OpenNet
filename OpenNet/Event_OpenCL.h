
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Event_OpenCL.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "Event.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Event_OpenCL : public Event
{

public:

    Event_OpenCL( bool aProfiling = false );

    // ===== Event ==========================================================
    virtual void Wait();

    cl_event mEvent;

}
