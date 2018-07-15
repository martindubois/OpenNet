
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/PacketSender.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class PacketSender
{

public:

    PacketSender(OpenNet::Adapter * aAdapter, unsigned int aPacketSize_byte, unsigned int aPacketQty);

    void Start();
    void Stop ();

// internal:

    unsigned int Run();

private:

    typedef enum
    {
        STATE_INIT    ,
        STATE_RUNNING ,
        STATE_STOPPING,

        STATE_QTY
    }
    State;

    OpenNet::Adapter * mAdapter        ;
    unsigned int       mPacketQty      ;
    unsigned int       mPacketSize_byte;
    State              mState          ;
    HANDLE             mThread         ;
    DWORD              mThreadId       ;

};
