
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Interface.h

#pragma once

// Constants / Constantes
/////////////////////////////////////////////////////////////////////////////

// {C0BE33A0-FFBA-46BA-B131-63BB331CA73E}
static const GUID OPEN_NET_DRIVER_INTERFACE = { 0xC0BE33A0, 0xFFBA, 0x46BA,{ 0xB1, 0x31, 0x63, 0xBB, 0x33, 0x1C, 0xA7, 0x3E } };

// ===== Adapter No / Numero d'adaptateur ===================================
#define  OPEN_NET_ADAPTER_NO_QTY (64)
#define  OPEN_NET_ADAPTER_NO_ANY (0xffffffff)

// ===== Adapter type / Type d'adaptateur ===================================
#define  OPEN_NET_ADAPTER_TYPE_INVALID  (0)
#define  OPEN_NET_ADAPTER_TYPE_ETHERNET (1)
#define  OPEN_NET_ADAPTER_TYPE_NDIS     (2)
#define  OPEN_NET_ADAPTER_TYPE_QTY      (3)
#define  OPEN_NET_ADAPTER_TYPE_ANY      (0xffffffff)

// ===== IoCtl ==============================================================

// Input   None
// Output  OpenNet_AdatperInfo
#define OPEN_NET_IOCTL_ADAPTER_INFO_GET    CTL_CODE( 0x8000, 0x800, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  OpenNet_AdapterStats
#define OPEN_NET_IOCTL_ADAPTER_STATS_GET   CTL_CODE( 0x8000, 0x810, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  None
#define OPEN_NET_IOCTL_ADAPTER_STATS_RESET CTL_CODE( 0x8000, 0x811, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   OpenNet_BufferInfo[ 1 .. N ]
// Output  None
#define OPEN_NET_IOCTL_BUFFER_QUEUE        CTL_CODE( 0x8000, 0x820, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  OpenNet_BufferInfp[ 1 .. N ]
#define OPEN_NET_IOCTL_BUFFER_RETRIEVE     CTL_CODE( 0x8000, 0x821, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   The paquet
// Output  None
#define OPEN_NET_IOCTL_PACKET_SEND         CTL_CODE( 0x8000, 0x830, METHOD_BUFFERED, FILE_ANY_ACCESS )

// Input   None
// Output  OpenNet_Version[ 0 .. N ]
#define OPEN_NET_IOCTL_VERSION_GET         CTL_CODE( 0x8000, 0x840, METHOD_BUFFERED, FILE_ANY_ACCESS )

// ===== Link state / Etat du lien ==========================================
#define OPEN_NET_LINK_STATE_INVALID (0)
#define OPEN_NET_LINK_STATE_DOWN    (1)
#define OPEN_NET_LINK_STATE_UP      (2)
#define OPEN_NET_LINK_STATE_QTY     (3)
#define OPRN_NET_LINK_STATE_ANY     (0xffffffff)

// ===== Packet state / Etat d'un paquet ====================================

// --> INVALID <--+<------------+<----------+
//        |       |             |           |
//        +--> RECEVING --> RECEIVED --> TO_SEND

#define OPEN_NET_PACKET_STATE_INVALID   (0)
#define OPNE_NET_PACKET_STATE_RECEIVED  (1)
#define OPEN_NET_PACKET_STATE_RECEIVING (2)
#define OPEN_NET_PACKET_STATE_TO_SEND   (3)
#define OPEN_NET_PACKET_STATE_QTY       (4)

// ===== Version ============================================================
#define OPEN_NET_VERSION_BUFFER_HEADER  (0)
#define OPEN_NET_VERSION_BUFFER_INFO    (1)

// Data type / Type de donnees
/////////////////////////////////////////////////////////////////////////////

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
    uint32_t mAdapterNo   ;
    uint32_t mAdapterType ;
    uint32_t mLinkState   ;
    uint32_t mPacketSize_byte;

    uint8_t mReserved0[112];

    char  mComment[128];
}
OpenNet_AdapterInfo;

/// \cond en
/// \brief  This structure is used to return the adapter's statistics.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour retourner les statistiques d'un
///         adaptateur.
/// \endcond
typedef struct
{
    uint64_t mRx_bytes;
    uint64_t mTx_bytes;

    uint32_t mAdapterStats_Get;
    uint32_t mRx_buffers      ;
    uint32_t mRx_errors       ;
    uint32_t mRx_interrupts   ;
    uint32_t mRx_packets      ;
    uint32_t mTx_buffers      ;
    uint32_t mTx_errors       ;
    uint32_t mTx_interrupt    ;
    uint32_t mTx_packets      ;
    uint32_t mTx_Send_bytes   ;
    uint32_t mTx_Send_errors  ;
    uint32_t mTx_Send_packets ;

    uint32_t mReserved0[113];

    uint32_t mAdapterStats_Reset;

    uint32_t mReserved1[127];
}
OpenNet_AdapterStats;

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

    uint32_t mPacketCount          ;
    uint32_t mPacketInfoOffset_byte;
    uint32_t mPacketOffset_byte    ;
    uint32_t mPacketSize_byte      ;

    uint8_t mReserved0[44];
}
OpenNet_BufferHeader;

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

    uint64_t mBuffer_PA;
    uint64_t mMarker_PA;
    uint32_t mSize_byte;

    uint8_t mReserved0[11];
}
OpenNet_BufferInfo;

/// \cond en
/// \brief  This structure is used to pass the information about a packet.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour passer les informations au sujet
///         d'un paquet.
/// \endcond
typedef struct
{
    uint64_t mToSendTo       ;
    uint32_t mPacketState    ;
    uint32_t mPacketSize_byte;

    uint8_t mReserved0[16];
}
OpenNet_PacketInfo;

/// \cond en
/// \brief  This structure is used to return the version of a component.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour retourner la version d'un
///         composant.
/// \endcond
typedef struct
{
    uint8_t mMajor;
    uint8_t mMinor;
    uint8_t mBuild;
    uint8_t mCompatibility;

    uint8_t mReserved0[128];

    char mComment[116];
    char mType   [  8];
}
OpenNet_Version;
