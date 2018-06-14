
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Interface.h

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

// {C0BE33A0-FFBA-46BA-B131-63BB331CA73E}
static const GUID OPEN_NET_DRIVER_INTERFACE = { 0xC0BE33A0, 0xFFBA, 0x46BA,{ 0xB1, 0x31, 0x63, 0xBB, 0x33, 0x1C, 0xA7, 0x3E } };

#define OPEN_NET_BUFFER_HEADER_VERSION  (0)
#define OPEN_NET_BUFFER_INFO_VERSION    (0)

// ===== Adapter numero =====================================================
#define OPEN_NET_ADAPTER_NO_QTY     (32)
#define OPEN_NET_ADAPTER_NO_UNKNOWN (99)

// ===== Adapter type =======================================================
#define  OPEN_NET_ADAPTER_TYPE_UNKNOWN  (0)
#define  OPEN_NET_ADAPTER_TYPE_ETHERNET (1)
#define  OPEN_NET_ADAPTER_TYPE_NDIS     (2)
#define  OPEN_NET_ADAPTER_TYPE_QTY      (3)

// ===== IoCtl ==============================================================

// Input   OpenNet_BufferInfo[ 1 .. N ]
// Output  None
#define OPEN_NET_IOCTL_BUFFER_QUEUE        CTL_CODE( 0x8000, 0x800, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  OpenNet_BufferInfp[ 1 .. N ]
#define OPEN_NET_IOCTL_BUFFER_RETRIEVE     CTL_CODE( 0x8000, 0x801, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  OpenNet_Config
#define OPEN_NET_IOCTL_CONFIG_GET          CTL_CODE( 0x8000, 0x810, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   OpenNet_Config
// Output  OpenNet_Config
#define OPEN_NET_IOCTL_CONFIG_SET          CTL_CODE( 0x8000, 0x811, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   OpenNet_AdapterConnect_In
// Output  None
#define OPEN_NET_IOCTL_CONNECT             CTL_CODE( 0x8000, 0x820, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Output  OpenNet_AdatperInfo
#define OPEN_NET_IOCTL_INFO_GET            CTL_CODE( 0x8000, 0x830, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   The paquet
// Output  None
#define OPEN_NET_IOCTL_PACKET_SEND         CTL_CODE( 0x8000, 0x840, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None// Input   None
// Output  OpenNet_State
#define OPEN_NET_IOCTL_STATE_GET           CTL_CODE( 0x8000, 0x850, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  OpenNet_AdapterStats
#define OPEN_NET_IOCTL_STATS_GET           CTL_CODE( 0x8000, 0x860, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  None
#define OPEN_NET_IOCTL_STATS_RESET         CTL_CODE( 0x8000, 0x861, METHOD_BUFFERED, FILE_ANY_ACCESS )

// ===== Link state =========================================================
#define OPEN_NET_LINK_STATE_UNKNOWN (0)
#define OPEN_NET_LINK_STATE_DOWN    (1)
#define OPEN_NET_LINK_STATE_UP      (2)
#define OPEN_NET_LINK_STATE_QTY     (3)

// ===== Mode ===============================================================
#define OPEN_NET_MODE_UNKNOWN     (0)
#define OPEN_NET_MODE_NORMAL      (1)
#define OPEN_NET_MODE_PROMISCUOUS (2)
#define OPEN_NET_MODE_QTY         (3)

// ===== Packet size ========================================================
#define OPEN_NET_PACKET_SIZE_MAX_byte (16384)
#define OPEN_NET_PACKET_SIZE_MIN_byte ( 1536)

// Data types
/////////////////////////////////////////////////////////////////////////////

/// \cond en
/// \brief  This structure is used for EthernetAddress.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour les adresse Ethernet.
/// \endcond
typedef struct
{
    uint8_t mAddress[6];

    uint8_t mReserved[2];
}
OpenNet_EthernetAddress;

/// \cond en
/// \brief  This structure is used to return the version of a component.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour retourner la version d'un
///         composant.
/// \endcond
typedef struct
{
    uint8_t mMajor        ;
    uint8_t mMinor        ;
    uint8_t mBuild        ;
    uint8_t mCompatibility;

    uint8_t mReserved0[44];

    char mComment[64];
    char mType   [16];
}
OpenNet_VersionInfo;

/// \cond en
/// \brief  This structure is used to pass the information about a buffer.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour passer les informations au sujet
///         d'un espace memoire.
/// \endcond
typedef struct
{
    uint32_t mVersion;

    uint8_t mReserved0[4];

    uint64_t mBuffer_PA;
    uint64_t mMarker_PA;
    uint32_t mSize_byte;

    uint8_t mReserved1[36];
}
OpenNet_BufferInfo;

/// \cond en
/// \brief  This structure is used to pass the configuration.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour passer la configuration.
/// \endcond
typedef struct
{
    uint32_t mMode           ;
    uint32_t mPacketSize_byte;

    uint32_t mReserved0[508];

    OpenNet_EthernetAddress  mEthernetAddress[50];

    uint8_t  mReserved1[112];
}
OpenNet_Config;

/// \cond en
/// \brief  This structure is used to pass the connection information.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour passer les information de
///         connexion.
/// \endcond
typedef struct
{
    uint64_t mEvent       ;
    void *   mSharedMemory;

    uint8_t  mReserved0[48];
}
OpenNet_Connect;

/// \cond en
/// \brief  This structure is used to return the information about an
///         adapter.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour retourner les information au
///         sujet d'un adaptateur.
/// \endcond
typedef struct
{
    uint32_t mAdapterType          ;
    uint32_t mCommonBufferSize_byte;
    uint32_t mPacketSize_byte      ;
    uint32_t mRx_Descriptors       ;
    uint32_t mTx_Descriptors       ;

    uint8_t mReserved0[100];

    OpenNet_EthernetAddress mEthernetAddress;

    char  mComment[128];

    OpenNet_VersionInfo mVersion_Driver  ;
    OpenNet_VersionInfo mVersion_Hardware;
    OpenNet_VersionInfo mVersion_ONK_Lib ;
}
OpenNet_Info;

/// \cond en
/// \brief  This structure is used to return the status.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour retourner l'etat.
/// \endcond
typedef struct
{
    struct
    {
        unsigned mFullDuplex : 1;
        unsigned mLinkUp     : 1;
        unsigned mTx_Off     : 1;

        unsigned mReserved0 : 29;
    }
    mFlags;

    uint32_t mAdapterNo ;
    uint32_t mSpeed_MB_s;

    uint32_t mReserved0[126];
}
OpenNet_State;

/// \cond en
/// \brief  This structure is used to return the adapter's statistics.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour retourner les statistiques d'un
///         adaptateur.
/// \endcond
typedef struct
{
    uint64_t mRx_byte;
    uint64_t mTx_byte;

    uint32_t mReserved0[12];

    uint32_t mIoCtl         ;
    uint32_t mStats_Get     ;
    uint32_t mRx_buffer     ;
    uint32_t mRx_error      ;
    uint32_t mRx_interrupt  ;
    uint32_t mRx_packet     ;
    uint32_t mTx_buffer     ;
    uint32_t mTx_error      ;
    uint32_t mTx_interrupt  ;
    uint32_t mTx_packet     ;
    uint32_t mTx_Send_byte  ;
    uint32_t mTx_Send_error ;
    uint32_t mTx_Send_packet;

    uint32_t mReserved1[99];

    uint32_t mIoCtl_Last       ;
    uint32_t mIoCtl_Last_Result;
    uint32_t mStats_Reset      ;

    uint32_t mReserved2[125];
}
OpenNet_Stats;
