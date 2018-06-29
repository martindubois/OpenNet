
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNetK/IoCtl.h

// TODO  Common.IoCtl
//       Retirer le IOCTL_STATS_RESET

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

// ===== IoCtl ==============================================================

// Input   None
// Output  OpenNet_Config
#define IOCTL_CONFIG_GET  CTL_CODE( 0x8000, 0x800, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   OpenNet_Config
// Output  OpenNet_Config
#define IOCTL_CONFIG_SET  CTL_CODE( 0x8000, 0x801, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   IoCtl_Connect_In
// Output  None
#define IOCTL_CONNECT     CTL_CODE( 0x8000, 0x810, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Output  OpenNet_AdatperInfo
#define IOCTL_INFO_GET    CTL_CODE( 0x8000, 0x820, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   The paquet
// Output  None
#define IOCTL_PACKET_SEND CTL_CODE( 0x8000, 0x830, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   IoCtl_BufferInfo[ 1 .. N ]
// Output  None
#define IOCTL_START       CTL_CODE( 0x8000, 0x840, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None// Input   None
// Output  OpenNet_State
#define IOCTL_STATE_GET   CTL_CODE( 0x8000, 0x850, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   IoCtl_Stats_Get_In
// Output  OpenNet_AdapterStats
#define IOCTL_STATS_GET   CTL_CODE( 0x8000, 0x860, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  None
#define IOCTL_STATS_RESET CTL_CODE( 0x8000, 0x861, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  None
#define IOCTL_STOP        CTL_CODE( 0x8000, 0x870, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    uint64_t mEvent       ;
    void *   mSharedMemory;
    uint32_t mSystemId    ;

    uint8_t  mReserved0[44];
}
IoCtl_Connect_In;

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
