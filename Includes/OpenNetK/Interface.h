
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Interface.h

#pragma once

// TODO  OpenNetK.Interface  Move the OPEN_NET_ADAPTER...,
//                           OPEN_NET_ADAPTER_TYPE_... and
//                           OPEN_NET_PACKET_SIZE_... constants to
//                           the file Common/OpenNetK/Constants.h

// Constants
/////////////////////////////////////////////////////////////////////////////

// {C0BE33A0-FFBA-46BA-B131-63BB331CA73E}
static const GUID OPEN_NET_DRIVER_INTERFACE = { 0xC0BE33A0, 0xFFBA, 0x46BA,{ 0xB1, 0x31, 0x63, 0xBB, 0x33, 0x1C, 0xA7, 0x3E } };

// ===== Adapter numero =====================================================
#define OPEN_NET_ADAPTER_NO_QTY     (32)
#define OPEN_NET_ADAPTER_NO_UNKNOWN (99)

// ===== Adapter type =======================================================
#define  OPEN_NET_ADAPTER_TYPE_UNKNOWN  (0)
#define  OPEN_NET_ADAPTER_TYPE_ETHERNET (1)
#define  OPEN_NET_ADAPTER_TYPE_NDIS     (2)
#define  OPEN_NET_ADAPTER_TYPE_QTY      (3)

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
    uint64_t mBuffer_PA;
    uint64_t mMarker_PA;
    uint32_t mPacketQty;
    uint32_t mSize_byte;

    uint8_t mReserved1[40];
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

// TODO  OpenNetK.Interface  Add queued buffer state

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

    uint32_t mAdapterNo  ;
    uint32_t mBufferCount;
    uint32_t mSpeed_MB_s ;
    uint32_t mSystemId   ;

    uint32_t mReserved0[123];
}
OpenNet_State;

// TODO  OpenNetK.Interface  Add statistics about kernel execution time.

/// \cond en
/// \brief  This structure is used to return the adapter's statistics.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour retourner les statistiques d'un
///         adaptateur.
/// \endcond
typedef struct
{
    uint32_t mBuffers_Process   ; //  0
    uint32_t mBuffer_InitHeader ;
    uint32_t mBuffer_Queue      ;
    uint32_t mBuffer_Receive    ;
    uint32_t mBuffer_Send       ;
    uint32_t mBuffer_SendPackets; //  5
    uint32_t mIoCtl             ;
    uint32_t mIoCtl_Config_Get  ;
    uint32_t mIoCtl_Config_Set  ;
    uint32_t mIoCtl_Connect     ;
    uint32_t mIoCtl_Info_Get    ; // 10
    uint32_t mIoCtl_Packet_Send ;
    uint32_t mIoCtl_Start       ;
    uint32_t mIoCtl_State_Get   ;
    uint32_t mIoCtl_Stats_Get   ;
    uint32_t mIoCtl_Stop        ; // 15
    uint32_t mTx_Packet         ;

    uint32_t mReserved0[110];
}
OpenNet_Stats_Adapter;

/// \cond en
/// \brief  This structure is used to return the adapter's statistics.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour retourner les statistiques d'un
///         adaptateur.
/// \endcond
typedef struct
{
    uint32_t mIoCtl_Last           ; //  0
    uint32_t mIoCtl_Last_Result    ;
    uint32_t mIoCtl_Stats_Get_Reset;
    uint32_t mIoCtl_Stats_Reset    ;

    uint32_t mReserved0[124];
}
OpenNet_Stats_Adapter_NoReset;

/// \cond en
/// \brief  This structure is used to return the adapter's statistics.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour retourner les statistiques d'un
///         adaptateur.
/// \endcond
typedef struct
{
    uint32_t mD0_Entry          ; //  0
    uint32_t mD0_Exit           ;
    uint32_t mInterrupt_Disable ;
    uint32_t mInterrupt_Enable  ;
    uint32_t mInterrupt_Process ;
    uint32_t mInterrupt_Process2; //  5
    uint32_t mPacket_Receive    ;
    uint32_t mPacket_Send       ;
    uint32_t mRx_Packet         ;
    uint32_t mSetConfig         ;
    uint32_t mStats_Get         ; // 10
    uint32_t mTx_Packet         ;

    uint32_t mReserved0[104];

    uint32_t mRx_BmcManagementDropper_packet     ;
    uint32_t mRx_CircuitBreakerDropped_packet    ;
    uint32_t mRx_LengthErrors_packet             ;
    uint32_t mRx_ManagementDropped_packet        ;
    uint32_t mRx_Missed_packet                   ; // 120
    uint32_t mRx_NoBuffer_packet                 ;
    uint32_t mRx_Oversize_packet                 ;
    uint32_t mRx_Undersize_packet                ;
    uint32_t mTx_DeferEvents                     ;
    uint32_t mTx_Discarded_packet                ; // 125
    uint32_t mTx_NoCrs_packet                    ;
    uint32_t mTx_HostCircuitBreakerDropped_packet; // 127
}
OpenNet_Stats_Hardware;

/// \cond en
/// \brief  This structure is used to return the adapter's statistics.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour retourner les statistiques d'un
///         adaptateur.
/// \endcond
typedef struct
{
    uint32_t mInterrupt_Process_Last_MessageId; // 0
    uint32_t mStats_Get_Reset                 ;
    uint32_t mStats_Reset                     ;

    uint32_t mReserved0[125];
}
OpenNet_Stats_Hardware_NoReset;

/// \cond en
/// \brief  This structure is used to return the adapter's statistics.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour retourner les statistiques d'un
///         adaptateur.
/// \endcond
typedef struct
{
    OpenNet_Stats_Adapter          mAdapter         ;
    OpenNet_Stats_Adapter_NoReset  mAdapter_NoReset ;
    OpenNet_Stats_Hardware         mHardware        ;
    OpenNet_Stats_Hardware_NoReset mHardware_NoReset;
}
OpenNet_Stats;
