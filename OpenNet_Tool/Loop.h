
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/Loop.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Function_Forward.h>
#include <OpenNet/System.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class Loop
{

public:

    //                Ethernet   Internal
    //
    // CIRCLE_FULL      +---->   -----+
    //                  | +--= 0 <--+ |
    //                  | |         | |
    //                  | +--> 1 ---+ |
    //                  +----=   <----+
    //
    // CIRCLE_HALF
    //                    +--= 0 <--+
    //                    |         |
    //                    +--> 1 ---+
    //
    // MIRROR_DOUBLE    +----> M
    //                  | +--= 0
    //                  | |
    //                  | +--> 1
    //                  +----= M
    //
    // MIRROR_SINGLE    +----> M
    //                  | +--= 0
    //                  | |
    //                  | +--> 1
    //                  +----- M
    //
    // MODE_EXPLOSION   +----> M -----+
    //                  | +--= 0 <--+ |
    //                  | |         | |
    //                  | +--> 1 ---+ |
    //                  +----= M <----+
    typedef enum
    {
        MODE_CIRCLE_FULL  ,
        MODE_CIRCLE_HALF  ,
        MODE_EXPLOSION    ,
        MODE_MIRROR_DOUBLE,
        MODE_MIRROR_SINGLE,

        MODE_QTY
    }
    Mode;

    Loop(unsigned int aBufferQty, unsigned int aPAcketSize_byte, unsigned int aPacketQty, Mode aMode);

    ~Loop();

    void DisplayAdapterStatistics();
    void DisplaySpeed            (double aDuration_s);

    void GetAdapterStatistics();

    void GetAndDisplayKernelStatistics();

    void ResetAdapterStatistics();

    void SendPackets();

    void Start();
    void Stop ();

    OpenNet::Adapter * mAdapters[2];

private:

    void Adapter_Connect();
    void Adapter_Get    ();

    void AddDestination();

    void ResetInputFilter();

    void SetConfig     ();
    void SetInputFilter();
    void SetProcessor  ();

    unsigned int              mBufferQty         ;
    OpenNet::Function_Forward mFunctions [2]     ;
    Mode                      mMode              ;
    unsigned int              mPacketQty         ;
    unsigned int              mPacketSize_byte   ;
    OpenNet::Processor      * mProcessor         ;
    unsigned int              mStatistics[2][128];
    OpenNet::System         * mSystem            ;

};
