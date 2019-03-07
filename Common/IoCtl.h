
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       Common/OpenNetK/IoCtl.h
//
// This file contains the definition of IoCtl code and the definition of data
// type used only to pass data in or out of IoCtl.

// TODO  Common.IoCtl
//       Normal (Feature) - Ajouter un IoCtl pour attendre sur un evennment
//       du pilote comme un buffer pret a traiter. Il faut aussi que ce IoCtl
//       retourne le nombre de fois que l'evenement est survenu depuis le
//       dernier appel. Cet IoCtl pourra etre utiliser pour le mode KERNEL de
//       CUDA.

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

// ===== 0.0.7 ==============================================================

// Input   None
// Output  None
#define IOCTL_PACKET_DROP                  IOCTL_CODE(50)

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
