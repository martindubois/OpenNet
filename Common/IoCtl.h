
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNetK/IoCtl.h

#pragma once

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    struct
    {
        unsigned mReset : 1;

        unsigned mReserved0 : 31;
    }
    mFlags;

    uint8_t mReserved0[60];
}
IoCtl_Stats_Get_In;
