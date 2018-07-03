
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/Loop.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Filter_Forward.h>
#include <OpenNet/System.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class Loop
{

public:

    typedef enum
    {
        MODE_DOUBLE_CIRCLE,
        MODE_DOUBLE_MIRROR,

        MODE_QTY
    }
    Mode;

    Loop(unsigned int aBufferQty, unsigned int aPAcketSize_byte, unsigned int aPacketQty, Mode aMode);

    ~Loop();

    void Display();

    void GetAndDisplayStatistics();

    void GetStatistics();

    void ResetStatistics();

    void SendPackets();

    void Start();
    void Stop ();

private:

    void Adapter_Connect();
    void Adapter_Get    ();

    void AddDestination();

    void ResetInputFilter();

    void SetConfig     ();
    void SetInputFilter();
    void SetProcessor  ();

    OpenNet::Adapter      * mAdapters  [2]     ;
    unsigned int            mBufferQty         ;
    OpenNet::Filter_Forward mFilters   [2]     ;
    Mode                    mMode              ;
    unsigned int            mPacketQty         ;
    unsigned int            mPacketSize_byte   ;
    OpenNet::Processor    * mProcessor         ;
    unsigned int            mStatistics[2][128];
    OpenNet::System       * mSystem            ;

};
