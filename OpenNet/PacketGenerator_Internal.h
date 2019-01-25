
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/PacketGenerator_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import =============================================================
#include <KmsLib/DebugLog.h>

// ===== Includes ===========================================================
#include <OpenNet/PacketGenerator.h>
#include <OpenNetK/PacketGenerator_Types.h>

// ===== OpenNet ============================================================
#include "Adapter_Internal.h"

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

private:

    void            Config_Apply   (const Config & aConfig);
    void            Config_Update  ();
    OpenNet::Status Config_Validate(const Config & aConfig);

    unsigned int Packet_Copy         (unsigned int aOffset, const void * aIn, unsigned int aInSize_byte);
    unsigned int Packet_Write16      (unsigned int aOffset, uint16_t aValue);
    unsigned int Packet_Write8       (unsigned int aOffset, uint8_t  aValue);
    unsigned int Packet_WriteEthernet(const OpenNet::Adapter::Info & aInfo, uint16_t aProtocol);
    unsigned int Packet_WriteIPv4    (unsigned int aOffset, uint8_t aProtocol);
    unsigned int Packet_WriteIPv4_UDP(unsigned int aOffset);

    void UpdateDriverConfig();
    void UpdatePacket      ();

    Adapter_Internal * mAdapter ;
    Config             mConfig  ;
    KmsLib::DebugLog   mDebugLog;
    bool               mRunning ;

    OpenNetK::PacketGenerator_Config  mDriverConfig;

};
