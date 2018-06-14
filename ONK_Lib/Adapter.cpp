
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Lib/Adapter.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== WDM ================================================================

#define INITGUID

#include <ntddk.h>

// ===== WDF ================================================================
#include <wdf.h>

// ===== Includes ===========================================================
#include <OpenNetK/StdInt.h>

#include <OpenNetK/Hardware.h>

#include <OpenNetK/Adapter.h>

// ===== Common =============================================================
#include "../Common/Version.h"

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void Adapter::Init()
    {
        memset(&mStats, 0, sizeof(mStats));

        mHardware = NULL;
    }

    void Adapter::SetHardware(Hardware * aHardware)
    {
        ASSERT(NULL != aHardware);

        ASSERT(NULL == mHardware);

        mHardware = aHardware;
    }

    int Adapter::IoCtl(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte)
    {
        int lResult;

        switch (aCode)
        {
        case OPEN_NET_IOCTL_BUFFER_QUEUE   : lResult = IoCtl_Buffer_Queue   (reinterpret_cast<const OpenNet_BufferInfo *>(aIn ), aInSize_byte ); break;
        case OPEN_NET_IOCTL_BUFFER_RETRIEVE: lResult = IoCtl_Buffer_Retrieve(reinterpret_cast<      OpenNet_BufferInfo *>(aOut), aOutSize_byte); break;
        case OPEN_NET_IOCTL_CONFIG_GET     : lResult = IoCtl_Config_Get     (reinterpret_cast<      OpenNet_Config     *>(aOut), aOutSize_byte); break;
        case OPEN_NET_IOCTL_CONFIG_SET     : lResult = IoCtl_Config_Set     (reinterpret_cast<const OpenNet_Config     *>(aIn ), aInSize_byte, reinterpret_cast<OpenNet_Config *>(aOut), aOutSize_byte); break;
        case OPEN_NET_IOCTL_CONNECT        : lResult = IoCtl_Connect        (reinterpret_cast<const OpenNet_Connect    *>(aIn ), aInSize_byte ); break;
        case OPEN_NET_IOCTL_INFO_GET       : lResult = IoCtl_Info_Get       (reinterpret_cast<      OpenNet_Info       *>(aOut), aOutSize_byte); break;
        case OPEN_NET_IOCTL_PACKET_SEND    : lResult = IoCtl_Packet_Send    (                                             aIn  , aInSize_byte ); break;
        case OPEN_NET_IOCTL_STATE_GET      : lResult = IoCtl_State_Get      (reinterpret_cast<      OpenNet_State      *>(aOut), aOutSize_byte); break;
        case OPEN_NET_IOCTL_STATS_GET      : lResult = IoCtl_Stats_Get      (reinterpret_cast<      OpenNet_Stats      *>(aOut), aOutSize_byte); break;
        case OPEN_NET_IOCTL_STATS_RESET    : lResult = IoCtl_Stats_Reset    (); break;

        default: lResult = -1;
        }

        mStats.mIoCtl++;
        mStats.mIoCtl_Last        = aCode  ;
        mStats.mIoCtl_Last_Result = lResult;

        return lResult;
    }

    unsigned int Adapter::IoCtl_InSize_GetMin(unsigned int aCode) const
    {
        unsigned int lResult;

        switch (aCode)
        {
        case OPEN_NET_IOCTL_BUFFER_QUEUE: lResult = sizeof(OpenNet_BufferInfo); break;
        case OPEN_NET_IOCTL_CONFIG_SET  : lResult = sizeof(OpenNet_Config    ); break;
        case OPEN_NET_IOCTL_CONNECT     : lResult = sizeof(OpenNet_Connect   ); break;

        default: lResult = 0;
        }

        return lResult;
    }

    unsigned int Adapter::IoCtl_OutSize_GetMin(unsigned int aCode) const
    {
        unsigned int lResult;

        switch (aCode)
        {
        case OPEN_NET_IOCTL_BUFFER_RETRIEVE: lResult = sizeof(OpenNet_BufferInfo); break;
        case OPEN_NET_IOCTL_CONFIG_GET     : lResult = sizeof(OpenNet_Config    ); break;
        case OPEN_NET_IOCTL_CONFIG_SET     : lResult = sizeof(OpenNet_Config    ); break;
        case OPEN_NET_IOCTL_INFO_GET       : lResult = sizeof(OpenNet_Info      ); break;
        case OPEN_NET_IOCTL_STATE_GET      : lResult = sizeof(OpenNet_State     ); break;
        case OPEN_NET_IOCTL_STATS_GET      : lResult = sizeof(OpenNet_Stats     ); break;

        default: lResult = 0;
        }

        return lResult;
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    // TODO  Test
    int Adapter::IoCtl_Buffer_Queue(const OpenNet_BufferInfo * aIn, unsigned int aInSize_byte)
    {
        ASSERT(NULL                       != aIn         );
        ASSERT(sizeof(OpenNet_BufferInfo) <= aInSize_byte);

        // TODO Dev

        return 0;
    }

    // TODO  Test
    int Adapter::IoCtl_Buffer_Retrieve(OpenNet_BufferInfo * aOut, unsigned int aOutSize_byte)
    {
        ASSERT(NULL                       != aOut         );
        ASSERT(sizeof(OpenNet_BufferInfo) <= aOutSize_byte);

        unsigned int lCount = aOutSize_byte / sizeof(OpenNet_BufferInfo);

        // TODO Dev

        return (lCount * sizeof(OpenNet_BufferInfo));
    }

    int Adapter::IoCtl_Config_Get(OpenNet_Config * aOut, unsigned int aOutSize_byte)
    {
        ASSERT(NULL                   != aOut         );
        ASSERT(sizeof(OpenNet_Config) <= aOutSize_byte);

        ASSERT(NULL != mHardware);

        mHardware->GetConfig(aOut);

        return sizeof(OpenNet_Config);
    }

    int Adapter::IoCtl_Config_Set(const OpenNet_Config * aIn, unsigned int aInSize_byte, OpenNet_Config * aOut, unsigned int aOutSize_byte)
    {
        ASSERT(NULL                   != aIn          );
        ASSERT(sizeof(OpenNet_Config) <= aInSize_byte );
        ASSERT(NULL                   != aOut         );
        ASSERT(sizeof(OpenNet_Config) <= aOutSize_byte);

        ASSERT(NULL != mHardware);

        mHardware->SetConfig(*aIn);
        mHardware->GetConfig(aOut);

        return sizeof(OpenNet_Config);
    }

    // TODO  Test
    int Adapter::IoCtl_Connect(const OpenNet_Connect * aIn, unsigned int aInSize_byte)
    {
        ASSERT(NULL != aIn         );
        ASSERT(   1 <= aInSize_byte);

        // TODO Dev

        return 0;
    }

    int Adapter::IoCtl_Info_Get(OpenNet_Info * aOut, unsigned int aOutSize_byte) const
    {
        ASSERT(NULL                 != aOut         );
        ASSERT(sizeof(OpenNet_Info) <= aOutSize_byte);

        ASSERT(NULL != mHardware);

        mHardware->GetInfo(aOut);

        return sizeof(OpenNet_Info);
    }

    int Adapter::IoCtl_Packet_Send(const void * aIn, unsigned int aInSize_byte)
    {
        ASSERT(NULL != mHardware);

        return ( mHardware->Packet_Send(aIn, aInSize_byte) ? 0 : -1 );
    }

    int Adapter::IoCtl_State_Get(OpenNet_State * aOut, unsigned int aOutSize_byte)
    {
        ASSERT(NULL                  != aOut         );
        ASSERT(sizeof(OpenNet_State) <= aOutSize_byte);

        ASSERT(NULL != mHardware);

        memset(aOut, 0, sizeof(OpenNet_State));

        mHardware->GetState(aOut);

        return sizeof(OpenNet_State);
    }

    int Adapter::IoCtl_Stats_Get(OpenNet_Stats * aOut, unsigned int aOutSize_byte) const
    {
        ASSERT(NULL           != aOut         );
        ASSERT(sizeof(mStats) <= aOutSize_byte);

        memcpy(aOut, &mStats, sizeof(mStats));

        mStats.mStats_Get++;
        
        return sizeof(mStats);
    }

    int Adapter::IoCtl_Stats_Reset()
    {
        memset(&mStats, 0, sizeof(mStats) / 2);

        mStats.mStats_Reset++;

        return 0;
    }

}
