
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       Common/OpenNetK/IoCtl.h
//
// This file contains the definition of IoCtl code and the definition of data
// type used only to pass data in or out of IoCtl.

// TODO  Common.IoCtl
//       Low (Cleanup) - Retirer le IOCTL_STATISTICS_RESET

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

#if defined( _KMS_LINUX_ ) && ! defined( _KMS_DRIVER_ )
    // ===== System =========================================================
    #include <sys/ioctl.h>
#endif

// Macros
/////////////////////////////////////////////////////////////////////////////

#ifdef _KMS_LINUX_
    #define IOCTL_CODE(N)      _IO('O',N)
    #define IOCTL_CODE_R(N,S)  _IOR('O',N,S)
    #define IOCTL_CODE_RW(N,S) _IOWR('O',N,S)
    #define IOCTL_CODE_W(N,S)  _IOW('O',N,S)
    #define IOCTL_CODE_W2(N,S) _IOW_BAD('O',N,S)
#endif

#ifdef _KMS_WINDOWS_
    #define IOCTL_CODE(N)      CTL_CODE( 0x8000, 0x800 + N, METHODE_BUFFERED, FILE_ANY_ACCESS)
    #define IOCTL_CODE_R(N,S)  CTL_CODE( 0x8000, 0x800 + N, METHODE_BUFFERED, FILE_ANY_ACCESS)
    #define IOCTL_CODE_RW(N,S) CTL_CODE( 0x8000, 0x800 + N, METHODE_BUFFERED, FILE_ANY_ACCESS)
    #define IOCTL_CODE_W(N,S)  CTL_CODE( 0x8000, 0x800 + N, METHODE_BUFFERED, FILE_ANY_ACCESS)
    #define IOCTL_CODE_W2(N,S) CTL_CODE( 0x8000, 0x800 + N, METHODE_BUFFERED, FILE_ANY_ACCESS)
#endif

// Constants
/////////////////////////////////////////////////////////////////////////////

// ===== IoCtl ==============================================================

// Input   None
// Output  OpenNetK::Adapter_Config
#define IOCTL_CONFIG_GET        IOCTL_CODE_R(0,OpenNetK::Adapter_Config)

// Input   OpenNetK::Adapter_Config
// Output  OpenNetK::Adapter_Config
#define IOCTL_CONFIG_SET        IOCTL_CODE_RW(1, OpenNetK::Adapter_Config)

// Input   IoCtl_Connect_In
// Output  None
#define IOCTL_CONNECT           IOCTL_CODE_W(16, IoCtl_Connect_In)

// Input   None
// Output  OpenNetK::Adatper_Info
#define IOCTL_INFO_GET          IOCTL_CODE_R(32, OpenNetK::Adapter_Info)

// Input   The paquet
// Output  None
#define IOCTL_PACKET_SEND       IOCTL_CODE_W2(48, 16 * 1024)

// Input   IoCtl_Packet_Send_Ex_In
//         The packet
// Output  None
#define IOCTL_PACKET_SEND_EX    IOCTL_CODE_W2(49, sizeof(IoCtl_Packet_Send_Ex_In) + (16 * 1024))

// Input   OpenNetK::Buffer[ 1 .. N ]
// Output  None
#define IOCTL_START             IOCTL_CODE_W(64, OpenNetK::Buffer[128])

// Input   None
// Output  OpenNetK::Adapter_State
#define IOCTL_STATE_GET         IOCTL_CODE_R(80, OpenNetK::Adapter_State)

// Input   IoCtl_Statistics_Get_In
// Output  uint32_t[ 0 .. N ]
#define IOCTL_STATISTICS_GET    IOCTL_CODE_RW(96, uint32_t[128])

// Input   None
// Output  None
#define IOCTL_STATISTICS_RESET  IOCTL_CODE(97)

// Input   None
// Output  None
#define IOCTL_STOP              IOCTL_CODE(112)

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
