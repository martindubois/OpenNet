
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All right reserved.
// Product    OpenNet
// File       ONK_Lib/Adapter_Linux.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/OS.h>
#include <OpenNetK/StdInt.h>

#include <OpenNetK/Adapter.h>
#include <OpenNetK/Hardware_Linux.h>

#include <OpenNetK/Adapter_Linux.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/IoCtl.h"

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void Adapter_Linux::Init( Adapter * aAdapter, OpenNetK_OSDep * aOSDep, void * aZone0 )
    {
        printk( KERN_DEBUG "%s( , , ,  )\n", __FUNCTION__ );

        ASSERT( NULL != aAdapter );
        ASSERT( NULL != aOSDep   );
        ASSERT( NULL != aZone0   );

        new ( & mZone0 ) SpinLock_Linux( aOSDep, aZone0 );

        aAdapter->Init( & mZone0 );
    }

}
