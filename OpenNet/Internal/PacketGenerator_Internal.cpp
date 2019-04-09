
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Internal/PacketGenerator_Internal.cpp

#define __CLASS__ "PacketGenerator_Internal::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== C ==================================================================
#include <memory.h>
#include <stdint.h>

#ifdef _KMS_LINUX_
    // ===== System =========================================================
    #include <sys/signal.h>
#endif

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>

// ===== Common =============================================================
#include "../Common/Constants.h"

// ===== OpenNet/Internal ===================================================

#include "../Constants.h"

#include "Adapter_Internal.h"

#include "PacketGenerator_Internal.h"

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static uint16_t Swap(uint16_t aIn);

// Public
/////////////////////////////////////////////////////////////////////////////

PacketGenerator_Internal::PacketGenerator_Internal() : mAdapter(NULL), mDebugLog( DEBUG_LOG_FOLDER, "PackeGenerator"), mRunning(false)
{
    // printf( __CLASS__ "PacketGenerator_Internal() - aThis = 0x%lx\n", reinterpret_cast< uint64_t >( this ) );

    memset(&mConfig                              ,    0, sizeof(mConfig                              ));
    memset(&mConfig.mDestinationEthernet.mAddress, 0xff, sizeof(mConfig.mDestinationEthernet.mAddress));
    memset(&mConfig.mDestinationIPv4             , 0xff, sizeof(mConfig.mDestinationIPv4             ));
    memset(&mDriverConfig                        ,    0, sizeof(mDriverConfig                        ));

    Config_Reset();
}

// ===== OpenNet::PacketGenerator ===========================================

PacketGenerator_Internal::~PacketGenerator_Internal()
{
    // printf( __CLASS__ "~PacketGenerator_Internal() - aThis = 0x%lx\n", reinterpret_cast< uint64_t >( this ) );

    try
    {
        if (mRunning)
        {
            assert(NULL != mAdapter);

            mAdapter->PacketGenerator_Stop();

            mRunning = false;
        }
    }
    catch (KmsLib::Exception * eE)
    {
        assert(NULL != eE);

        mDebugLog.LogTime();
        mDebugLog.Log(__FILE__, __CLASS__ "~PacketGenerator_Internal", __LINE__);
        mDebugLog.Log(eE);
    }
}

OpenNet::Status PacketGenerator_Internal::GetConfig(Config * aOut) const
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mConfig, sizeof(mConfig));

    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::ResetConfig()
{
    if (mRunning)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "ResetConfig", __LINE__);
        return OpenNet::STATUS_PACKET_GENERATOR_RUNNING;
    }

    try
    {
        Config_Reset();

        if (NULL != mAdapter)
        {
            UpdateDriverConfig();
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(eE);
        return OpenNet::STATUS_EXCEPTION;
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::SetAdapter(OpenNet::Adapter * aAdapter)
{
    if (NULL == aAdapter)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "SetAdapter", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (NULL != mAdapter)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "SetAdapter", __LINE__);
        return OpenNet::STATUS_ADAPTER_ALREADY_SET;
    }

    mAdapter = dynamic_cast<Adapter_Internal *>( aAdapter );
    if (NULL == mAdapter)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "SetAdapter", __LINE__);
        return OpenNet::STATUS_INVALID_ADAPTER;
    }

    try
    {
        UpdateDriverConfig();
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(eE);
        return OpenNet::STATUS_EXCEPTION;
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::SetConfig(const Config & aConfig)
{
    if (NULL == (&aConfig))
    {
        mDebugLog.Log(__FILE__, __CLASS__ "SetConfig", __LINE__);
        return OpenNet::STATUS_INVALID_REFERENCE;
    }

    OpenNet::Status lStatus = Config_Validate(aConfig);
    if (OpenNet::STATUS_OK != lStatus)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "SetConfig", __LINE__);
        return lStatus;
    }

    if (mRunning)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "SetConfig", __LINE__);
        return OpenNet::STATUS_PACKET_GENERATOR_RUNNING;
    }

    try
    {
        Config_Apply(aConfig);
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(eE);
        return OpenNet::STATUS_EXCEPTION;
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::Display(FILE * aOut)
{
    if (NULL == aOut)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Display", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "PacketGenerator :\n");
    fprintf(aOut, "  Adapter   = %s\n", (NULL == mAdapter) ? "Not set" : mAdapter->GetName());
    fprintf(aOut, "  Running   = %s\b", mRunning ? "true" : "false");
    
    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::Start()
{
    if (NULL == mAdapter)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Start", __LINE__);
        return OpenNet::STATUS_ADAPTER_NOT_SET;
    }

    if (mRunning)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Start", __LINE__);
        return OpenNet::STATUS_PACKET_GENERATOR_RUNNING;
    }

    try
    {
        mAdapter->PacketGenerator_Start();

        mRunning = true;
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Start", __LINE__);
        mDebugLog.Log(eE);
        return OpenNet::STATUS_EXCEPTION;
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status PacketGenerator_Internal::Stop()
{
    if (NULL == mAdapter)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Stop", __LINE__);
        return OpenNet::STATUS_ADAPTER_NOT_SET;
    }

    if (!mRunning)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Stop", __LINE__);
        return OpenNet::STATUS_PACKET_GENERATOR_STOPPED;
    }

    try
    {
        mRunning = false;

        mAdapter->PacketGenerator_Stop();
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Stop", __LINE__);
        mDebugLog.Log(eE);
        return OpenNet::STATUS_EXCEPTION;
    }

    return OpenNet::STATUS_OK;
}

// Private
/////////////////////////////////////////////////////////////////////////////

// aConfig [---;R--]
//
// Exception  KmsLib::Exception *  See UpdateDriverConfig
void PacketGenerator_Internal::Config_Apply(const Config & aConfig)
{
    assert(NULL != (&aConfig));

    memcpy(&mConfig, &aConfig, sizeof(mConfig));

    if (NULL != mAdapter)
    {
        UpdateDriverConfig();
    }
}

void PacketGenerator_Internal::Config_Reset()
{
    mConfig.mAllowedIndexRepeat = REPEAT_COUNT_MAX ;
    mConfig.mBandwidth_MiB_s    =              50.0;
    mConfig.mDestinationPort    = 0x0a0a           ;
    mConfig.mEthernetProtocol   = 0x0a0a           ;
    mConfig.mPacketSize_byte    =            1024  ;
    mConfig.mProtocol           = PROTOCOL_ETHERNET;
    mConfig.mSourcePort         = 0x0909           ;
}

void PacketGenerator_Internal::Config_Update()
{
    mConfig.mAllowedIndexRepeat = mDriverConfig.mAllowedIndexRepeat;
    mConfig.mIndexOffset_byte   = mDriverConfig.mIndexOffset_byte  ;
    mConfig.mPacketSize_byte    = mDriverConfig.mPacketSize_byte   ;
}

// aConfig [---;R--]
OpenNet::Status PacketGenerator_Internal::Config_Validate(const Config & aConfig)
{
    assert(NULL != (&aConfig));

    if (0.0 >= aConfig.mBandwidth_MiB_s)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Config_Validate", __LINE__);
        return OpenNet::STATUS_INVALID_BANDWIDTH;
    }

    if (aConfig.mPacketSize_byte < (aConfig.mIndexOffset_byte + sizeof(uint32_t)))
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Config_Validate", __LINE__);
        return OpenNet::STATUS_INVALID_OFFSET;
    }

    if (0 >= aConfig.mPacketSize_byte)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Config_Validate", __LINE__);
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    if (PACKET_SIZE_MAX_byte < aConfig.mPacketSize_byte)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Config_Validate", __LINE__);
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    if (PROTOCOL_QTY <= aConfig.mProtocol)
    {
        mDebugLog.Log(__FILE__, __CLASS__ "Config_Validate", __LINE__);
        return OpenNet::STATUS_INVALID_PROTOCOL;
    }

    return OpenNet::STATUS_OK;
}

// Exception  KmsLib::Exception *  See Adapter_Internal::PacketGenerator_GetConfig
//                                 and Adapter_Internal::PacketGenerator_SetConfig
void PacketGenerator_Internal::UpdateDriverConfig()
{
    assert(NULL                 != mAdapter                );
    assert(                 0.0 <  mConfig.mBandwidth_MiB_s);
    assert(                 0   <  mConfig.mPacketSize_byte);
    assert(PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte);

    mAdapter->PacketGenerator_GetConfig(&mDriverConfig);

    mDriverConfig.mAllowedIndexRepeat = mConfig.mAllowedIndexRepeat;
    mDriverConfig.mIndexOffset_byte   = mConfig.mIndexOffset_byte  ;
    mDriverConfig.mPacketPer100ms     = static_cast<uint32_t>( ( mConfig.mBandwidth_MiB_s * 1000000.0 / 9.0 ) / ( mConfig.mPacketSize_byte + 24 ) );
    mDriverConfig.mPacketSize_byte    = mConfig.mPacketSize_byte   ;

    UpdatePacket();

    mAdapter->PacketGenerator_SetConfig(&mDriverConfig);

    Config_Update();
}

// aOffset       Offset where to copy the data
// aIn [---;R--] Data to copy
// aInSize_byte  Size of the data to copy
//
// Return  This method returns the offset just after the copied data.
unsigned int PacketGenerator_Internal::Packet_Copy(unsigned int aOffset, const void * aIn, unsigned int aInSize_byte)
{
    assert(NULL != aIn         );
    assert(   0 <  aInSize_byte);

    unsigned int lResult = aOffset;

    memcpy(mDriverConfig.mPacket + lResult, aIn, aInSize_byte); lResult += aInSize_byte;

    return lResult;
}

// aOffset  Offset where to write the data
// aValue   The value to write
//
// Return  This method returns the offset just after the writen data.
unsigned int PacketGenerator_Internal::Packet_Write16(unsigned int aOffset, uint16_t aValue)
{
    assert(0 < aOffset);

    unsigned int lResult = aOffset;

    (*reinterpret_cast<uint32_t *>(mDriverConfig.mPacket + lResult)) = aValue; lResult += sizeof(aValue);

    return lResult;
}

// aOffset  Offset where to write the data
// aValue   The value to write
//
// Return  This method returns the offset just after the writen data.
unsigned int PacketGenerator_Internal::Packet_Write8(unsigned int aOffset, uint8_t aValue)
{
    assert(0 < aOffset);

    unsigned int lResult = aOffset;

    (*reinterpret_cast<uint32_t *>(mDriverConfig.mPacket + lResult)) = aValue; lResult += sizeof(aValue);

    return lResult;
}

// aInfo [---;R--] The information about the adapter
// aProtocol       The ethernet protocol in network order
//
// Return  This method returns the offset just after the writen data.
unsigned int PacketGenerator_Internal::Packet_WriteEthernet(const OpenNet::Adapter::Info & aInfo, uint16_t aProtocol)
{
    assert(NULL != (&aInfo));

    unsigned int lResult = 0;

    lResult = Packet_Copy   (lResult, mConfig.mDestinationEthernet.mAddress, sizeof(mConfig.mDestinationEthernet.mAddress));
    lResult = Packet_Copy   (lResult, aInfo  .mEthernetAddress    .mAddress, sizeof(aInfo  .mEthernetAddress    .mAddress));
    lResult = Packet_Write16(lResult, aProtocol);

    return lResult;
}

// aOffset    Offset where to write the data
// aProtocol  The IPv4 protocol
//
// Return  This method returns the offset just after the write data.
unsigned int PacketGenerator_Internal::Packet_WriteIPv4(unsigned int aOffset, uint8_t aProtocol)
{
    assert(0 < aOffset);

    assert(aOffset < mConfig.mPacketSize_byte);

    unsigned int lResult = aOffset;

    lResult = Packet_Write8 (lResult, 0x54);
    lResult = Packet_Write8 (lResult, 0x00);
    lResult = Packet_Write16(lResult, Swap(mConfig.mPacketSize_byte - aOffset));
    lResult = Packet_Write16(lResult, 0x0000);
    lResult = Packet_Write8 (lResult, 0x40);
    lResult = Packet_Write8 (lResult, 0x00);
    lResult = Packet_Write8 (lResult, 0x80);
    lResult = Packet_Write8 (lResult, aProtocol);
    lResult = Packet_Write16(lResult, 0x0000);
    lResult = Packet_Copy   (lResult, mConfig.mSourceIPv4     .mAddress, sizeof(mConfig.mSourceIPv4     .mAddress));
    lResult = Packet_Copy   (lResult, mConfig.mDestinationIPv4.mAddress, sizeof(mConfig.mDestinationIPv4.mAddress));

    return lResult;
}

// aOffset  Offset where to write the data
//
// Return  This method returns the offset just after the write data.
unsigned int PacketGenerator_Internal::Packet_WriteIPv4_UDP(unsigned int aOffset)
{
    assert(0 < aOffset);

    assert(aOffset < mConfig.mPacketSize_byte);

    unsigned int lResult = aOffset;

    lResult = Packet_Write16(lResult, mConfig.mSourcePort     );
    lResult = Packet_Write16(lResult, mConfig.mDestinationPort);
    lResult = Packet_Write16(lResult, mConfig.mPacketSize_byte - aOffset);
    lResult = Packet_Write16(lResult, 0x0000);

    return lResult;
}

void PacketGenerator_Internal::UpdatePacket()
{
    assert(NULL != mAdapter);

    OpenNet::Adapter::Info lInfo;

    OpenNet::Status lStatus = mAdapter->GetInfo(&lInfo);
    assert(OpenNet::STATUS_OK == lStatus);
    (void)(lStatus);

    unsigned int lOffset;

    switch (mConfig.mProtocol)
    {
    case PROTOCOL_ETHERNET:
        lOffset = Packet_WriteEthernet(lInfo, mConfig.mEthernetProtocol);
        break;

    case PROTOCOL_IPv4    :
        lOffset = Packet_WriteEthernet(lInfo, 0x0800);
        lOffset = Packet_WriteIPv4    (lOffset, mConfig.mIPv4Protocol);
        break;

    case PROTOCOL_IPv4_UDP:
        lOffset = Packet_WriteEthernet(lInfo, 0x0800);
        lOffset = Packet_WriteIPv4    (lOffset, 0x11);
        lOffset = Packet_WriteIPv4_UDP(lOffset);
        break;

    default: assert(false);
    }

    if ((0 != mDriverConfig.mIndexOffset_byte) && (mDriverConfig.mIndexOffset_byte < lOffset))
    {
        mDriverConfig.mIndexOffset_byte = lOffset;
    }
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aIn  The value to swap
//
// Return  The swapped value
uint16_t Swap(uint16_t aIn)
{
    return ((aIn << 8) || (aIn >> 8));
}