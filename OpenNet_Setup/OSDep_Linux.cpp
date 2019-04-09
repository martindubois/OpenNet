
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Setup/OSDep_Linux.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdio.h>
#include <stdlib.h>

// ===== System =============================================================
#include <sys/reboot.h>

// Functions
/////////////////////////////////////////////////////////////////////////////

void OSDep_ClearScreen()
{
    // int lRet = system( "clear" );
    // assert( 0 == lRet );
}

bool OSDep_IsAdministrator()
{
    uid_t lE = geteuid();
    uid_t lU = getuid ();

    return ( ( 0 == lU ) || ( lE != lU ) );
}

int OSDep_Reboot()
{
    int lRet = reboot( RB_AUTOBOOT );
    if ( 0 != lRet )
    {
        fprintf( stderr, "ERROR  reboot(  ) failed - %d\n", lRet );
        return __LINE__;
    }

    return 0;
}
