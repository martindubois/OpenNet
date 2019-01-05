
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNetK/IoCtl.h
//
// This file contains the definition of IoCtl code and the definition of data
// type used only to pass data in or out of IoCtl.

// TODO  Common.IoCtl
//       Low (Cleanup) - Retirer le IOCTL_STATISTICS_RESET

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

// ===== IoCtl ==============================================================

// Input   None
// Output  OpenNetK::Adapter_Config
#define IOCTL_CONFIG_GET        CTL_CODE( 0x8000, 0x800, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   OpenNetK::Adapter_Config
// Output  OpenNetK::Adapter_Config
#define IOCTL_CONFIG_SET        CTL_CODE( 0x8000, 0x801, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   IoCtl_Connect_In
// Output  None
#define IOCTL_CONNECT           CTL_CODE( 0x8000, 0x810, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Output  OpenNetK::Adatper_Info
#define IOCTL_INFO_GET          CTL_CODE( 0x8000, 0x820, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   The paquet
// Output  None
#define IOCTL_PACKET_SEND       CTL_CODE( 0x8000, 0x830, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   IoCtl_Packet_Send_Ex_In
//         The packet
// Output  None
#define IOCTL_PACKET_SEND_EX    CTL_CODE( 0x8000, 0x831, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   OpenNetK::Buffer[ 1 .. N ]
// Output  None
#define IOCTL_START             CTL_CODE( 0x8000, 0x840, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  OpenNetK::Adapter_State
#define IOCTL_STATE_GET         CTL_CODE( 0x8000, 0x850, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   IoCtl_Statistics_Get_In
// Output  uint32_t[ 0 .. N ]
#define IOCTL_STATISTICS_GET    CTL_CODE( 0x8000, 0x860, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  None
#define IOCTL_STATISTICS_RESET  CTL_CODE( 0x8000, 0x861, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  None
#define IOCTL_STOP              CTL_CODE( 0x8000, 0x870, METHOD_BUFFERED, FILE_ANY_ACCESS )

// TODO  OpenNetK.IoCtl
//       Normal - Ajouter PACKET_GENERATOR_START et PACKET_GENERATOR_STOP et
//       implanter un packet generator directement dans ONK_Lib

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
        unsigned mReserved0 : 32;
    }
    mFlags;

    uint32_t mRepeatCount;

    uint8_t mReserved0[24];
}
IoCtl_Packet_Send_Ex_In;

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
