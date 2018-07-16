
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/PacketGenerator_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import =============================================================
#include <KmsLib/DebugLog.h>

// ===== Common =============================================================
#include "../Common/OpenNet/PacketGenerator_Statistics.h"

// ===== Includes ===========================================================
#include <OpenNet/PacketGenerator.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class PacketGenerator_Internal : public OpenNet::PacketGenerator
{

public:

    PacketGenerator_Internal();

    // ===== OpenNet::PacketGenerator =======================================

    virtual ~PacketGenerator_Internal();

    virtual OpenNet::Status GetConfig (Config * aOut) const;
    virtual OpenNet::Status SetAdapter(OpenNet::Adapter * aAdapter);
    virtual OpenNet::Status SetConfig (const Config & aConfig);
    virtual OpenNet::Status Display   (FILE * aOut);
    virtual OpenNet::Status Start     ();
    virtual OpenNet::Status Stop      ();

    // ===== OpenNet::StatisticsProvider ====================================
    virtual OpenNet::Status GetStatistics  (unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);
    virtual OpenNet::Status ResetStatistics();

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

    OpenNet::Status Config_Apply   (const Config & aConfig);
    OpenNet::Status Config_Validate(const Config & aConfig);

    OpenNet::Adapter * mAdapter ;
    Config             mConfig  ;
    KmsLib::DebugLog   mDebugLog;
    State              mState   ;
    HANDLE             mThread  ;
    DWORD              mThreadId;

    unsigned int mStatistics[OpenNet::PACKET_GENERATOR_STATS_QTY];

};