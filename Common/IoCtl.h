
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       Common/OpenNetK/IoCtl.h
//
// This file contains the definition of IoCtl code and the definition of data
// type used only to pass data in or out of IoCtl.

// TODO  Common.IoCtl
//       Low (Cleanup) - Retirer IOCTL_STATISTICS_RESET

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
    // 3 2         1         0
    // 10987654321098765432109876543210
    // {}{--14 bits---}{8 bits}{8 bits}
    //  |      |          |        |
    //  |      |          |        +-- Function code (N)
    //  |      |          +-- Type - Alwais 'O' (0x4f)
    //  |      +-- Size in bytes (sizeof(S))
    //  +-- Direction - 00 = _IOC_NONE, 01 = _IOC_WRITE, 10 = _IOC_READ

    #define IOCTL_CODE(N)      _IO('O',N)
    #define IOCTL_CODE_R(N,S)  _IOR('O',N,S)
    #define IOCTL_CODE_RW(N,S) _IOWR('O',N,S)
    #define IOCTL_CODE_W(N,S)  _IOW('O',N,S)
    #define IOCTL_CODE_W2(N,S) _IOW_BAD('O',N,S)
#endif

#ifdef _KMS_WINDOWS_
    // 3 2         1         0
    // 10987654321098765432109876543210
    // |{---15 bits---}{}|{-11 bits-}{}
    // |       |        ||     |      |
    // |       |        ||     |      +-- Transfer type - 00 = METHOD_BUFFERED
    // |       |        ||     +-- Function code (N)
    // |       |        |+-- Custom - Always 1
    // |       |        +-- Required access - 00 = FILE_ANY_ACCESS
    // |       +-- Device type - 0
    // +-- Common - Always 1
    #define IOCTL_CODE(N)      CTL_CODE( 0x8000, 0x800 + N, METHOD_BUFFERED, FILE_ANY_ACCESS)
    #define IOCTL_CODE_R(N,S)  CTL_CODE( 0x8000, 0x800 + N, METHOD_BUFFERED, FILE_ANY_ACCESS)
    #define IOCTL_CODE_RW(N,S) CTL_CODE( 0x8000, 0x800 + N, METHOD_BUFFERED, FILE_ANY_ACCESS)
    #define IOCTL_CODE_W(N,S)  CTL_CODE( 0x8000, 0x800 + N, METHOD_BUFFERED, FILE_ANY_ACCESS)
    #define IOCTL_CODE_W2(N,S) CTL_CODE( 0x8000, 0x800 + N, METHOD_BUFFERED, FILE_ANY_ACCESS)
#endif

// Constants
/////////////////////////////////////////////////////////////////////////////

// ===== IoCtl ==============================================================

// ----- 0 - 15  Config -----------------------------------------------------

// Input   None
// Output  OpenNetK::Adapter_Config
#define IOCTL_CONFIG_GET        IOCTL_CODE_R(0,OpenNetK::Adapter_Config)

// Input   OpenNetK::Adapter_Config
// Output  OpenNetK::Adapter_Config
#define IOCTL_CONFIG_SET        IOCTL_CODE_RW(1, OpenNetK::Adapter_Config)

// ----- 16 - 31  Connect ---------------------------------------------------

// Input   IoCtl_Connect_In
// Output  IoCtl_Connect_Out
#define IOCTL_CONNECT           IOCTL_CODE_RW(16, IoCtl_Connect_In)

// ----- 32 - 47  Info ------------------------------------------------------

// Input   None
// Output  OpenNetK::Adatper_Info
#define IOCTL_INFO_GET          IOCTL_CODE_R(32, OpenNetK::Adapter_Info)

// ----- 48 - 63  Packet ----------------------------------------------------

// Input   IoCtl_Packet_Send_Ex_In
//         The packet
// Output  None
#define IOCTL_PACKET_SEND_EX    IOCTL_CODE_W2(49, sizeof(IoCtl_Packet_Send_Ex_In) + (16 * 1024))

// Input   None
// Output  None
#define IOCTL_PACKET_DROP                  IOCTL_CODE(50)

// ----- 64 - 79  Start -----------------------------------------------------

// Input   OpenNetK::Buffer[ 1 .. N ]
// Output  None
#define IOCTL_START             IOCTL_CODE_W(64, OpenNetK::Buffer[128])

// ----- 80 - 95  State -----------------------------------------------------

// Input   None
// Output  OpenNetK::Adapter_State
#define IOCTL_STATE_GET         IOCTL_CODE_R(80, OpenNetK::Adapter_State)

// ----- 96 - 111  Statistics -----------------------------------------------

// Input   IoCtl_Statistics_Get_In
// Output  uint32_t[ 0 .. N ]
#define IOCTL_STATISTICS_GET    IOCTL_CODE_RW(96, uint32_t[128])

// Input   None
// Output  None
#define IOCTL_STATISTICS_RESET  IOCTL_CODE(97)

// ----- 112 - 127  Stop ----------------------------------------------------

// Input   None
// Output  None
#define IOCTL_STOP              IOCTL_CODE(112)

// ----- 128 - 143  Packet Generator ----------------------------------------

// Input   None
// Output  OpenNetK::Generator_Config
#define IOCTL_PACKET_GENERATOR_CONFIG_GET  IOCTL_CODE_R(128, OpenNetK::PacketGenerator_Config)

// Input  OpenNetK::Generator_Config
// Outpt  OpenNetK::Generator_Config
#define IOCTL_PACKET_GENERATOR_CONFIG_SET  IOCTL_CODE_W(129, OpenNetK::PacketGenerator_Config)

// Input   None
// Output  None
#define IOCTL_PACKET_GENERATOR_START      IOCTL_CODE(130)

// Input   None
// Output  None
#define IOCTL_PACKET_GENERATOR_STOP       IOCTL_CODE(131)

// ----- 144 - 159  Tx ------------------------------------------------------

// Input   None
// Output  None
#define IOCTL_TX_DISABLE                  IOCTL_CODE(144)

// Input   None
// Output  None
#define IOCTL_TX_ENABLE                   IOCTL_CODE(145)

// ----- 160 - 175  Event ---------------------------------------------------

// Input   IoCtl_Event_Wait_In
// Output  OpenNetK::Event[ 0 .. N ]
#define IOCTL_EVENT_WAIT                  IOCTL_CODE_RW(160, OpenNetK::Event[32])

// Input   None
// Output  None
#define IOCTL_EVENT_WAIT_CANCEL           IOCTL_CODE(161)

// ----- 176 - 192  License -------------------------------------------------

// Input   IoCtl_License_Set_In
// Output  IoCtl_License_Set_Out
#define IOCTL_LICENSE_SET       IOCTL_CODE_RW(176, IoCtl_License_Set_In)

// Data types
/////////////////////////////////////////////////////////////////////////////

// mSharedMemory  The user space virtual address of the memory shared by all
//                adapter connected to a same system
// mSystemId      The process id of the process controlling the system
typedef struct
{
    uint8_t  mReserved0[ 8 ];

    void *   mSharedMemory;
    uint32_t mSystemId    ;

    uint8_t  mReserved1[44];
}
IoCtl_Connect_In;

// mAdapterNo  The numero of the adapter
typedef struct
{
    uint32_t mAdapterNo;

    uint8_t mReserved[60];
}
IoCtl_Connect_Out;

typedef struct
{
    uint32_t mOutputSize_byte;

    uint8_t mReserved0[60];
}
IoCtl_Event_Wait_In;

typedef struct
{
    uint32_t mKey;

    uint8_t mReserved0[60];
}
IoCtl_License_Set_In;

typedef struct
{
    struct
    {
        unsigned mLicenseOk : 1;

        unsigned mReserved0 : 31;
    }
    mFlags;

    uint8_t mReserved0[60];
}
IoCtl_License_Set_Out;

// mRepeatCount  The number of time the packet must be send
// mSize_byte    The size of packet
typedef struct
{
    struct
    {
        unsigned mReserved0 : 32;
    }
    mFlags;

    uint32_t mRepeatCount;
    uint16_t mSize_byte  ;

    uint8_t mReserved0[22];
}
IoCtl_Packet_Send_Ex_In;

// mFlags.mReset  When set to true, all the statisticas counters are reset to
//                0 just after the reading operation.
// mOutSize_byte  The size of the output buffer
typedef struct
{
    struct
    {
        unsigned mReset : 1;

        unsigned mReserved0 : 31;
    }
    mFlags;

    uint32_t mOutputSize_byte;

    uint8_t mReserved0[ 56 ];
}
IoCtl_Statistics_Get_In;
