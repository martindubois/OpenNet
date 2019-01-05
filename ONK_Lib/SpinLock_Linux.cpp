
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Lib/SpinLock_Linux.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/OS.h>
#include <OpenNetK/StdInt.h>

#include <OpenNetK/SpinLock_Linux.h>

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    SpinLock_Linux::SpinLock_Linux()
    {
    }

    // ===== SpinLock =======================================================

    // CRITICAL PATH - Buffer
    void SpinLock_Linux::Lock()
    {
    }

    // CRITICAL PATH - Buffer
    void SpinLock_Linux::Unlock()
    {
    }

};
