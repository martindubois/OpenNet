
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONL_Lib/Hardware_Linux.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/OS.h>
#include <OpenNetK/StdInt.h>

#include <OpenNetK/Hardware.h>

#include <OpenNetK/Hardware_Linux.h>

// Data type
/////////////////////////////////////////////////////////////////////////////

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void Hardware_Linux::Init( Hardware * aHardware, OpenNetK_OSDep * aOSDep, void * aZone0 )
    {
        ASSERT( NULL != aHardware );
        ASSERT( NULL != aOSDep    );
        ASSERT( NULL != aZone0    );

        new ( & mZone0 ) SpinLock_Linux( aOSDep, aZone0 );

        aHardware->Init( & mZone0 );
    }
    
}
