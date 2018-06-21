
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Types.h

#pragma once

// Constants / Constantes
/////////////////////////////////////////////////////////////////////////////

// ===== Buffer state =======================================================

// --> INVALID --> SENDING <---------------------------+
//                  |                                  |
//                  +--> RECEIVING --> PROCESSING --> PROCESSED

#define OPEN_NET_BUFFER_STATE_INVALID    (0)
#define OPEN_NET_BUFFER_STATE_PROCESSED  (1)
#define OPNE_NET_BUFFER_STATE_PROCESSING (2)
#define OPEN_NET_BUFFER_STATE_RECEIVING  (3)
#define OPEN_NET_BUFFER_STATE_SENDING    (4)
#define OPEN_NET_BUFFER_STATE_QTY        (5)

// ===== Packet state / Etat d'un paquet ====================================

// --> INVALID --> SENDING <------------------------+
//                  |                               |
//                  +--> RECEVING --> RECEIVED --> PROCESSED

#define OPEN_NET_PACKET_STATE_INVALID    (0)
#define OPEN_NET_PACKET_STATE_PROCESSED  (1)
#define OPEN_NET_PACKET_STATE_RECEIVED   (2)
#define OPEN_NET_PACKET_STATE_RECEIVING  (3)
#define OPEN_NET_PACKET_STATE_SENDING    (4)
#define OPEN_NET_PACKET_STATE_QTY        (5)

// Data type / Type de donnees
/////////////////////////////////////////////////////////////////////////////

/// \cond en
/// \brief  This structure is used to pass the information about a buffer.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour passer les informations au sujet
///         d'un espace memoire.
/// \endcond
typedef struct
{
    uint32_t mPacketQty            ;
    uint32_t mPacketInfoOffset_byte;
    uint32_t mPacketSize_byte      ;
    uint32_t mBufferState          ;

    uint8_t mReserved0[48];
}
OpenNet_BufferHeader;

/// \cond en
/// \brief  This structure is used to pass the information about a packet.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour passer les informations au sujet
///         d'un paquet.
/// \endcond
typedef struct
{
    uint32_t mToSendTo;

    uint8_t mReserved0[4];

    uint32_t mPacketOffset_byte;
    uint32_t mPacketState      ;
    uint32_t mPacketSize_byte  ;

    uint8_t mReserved1[12];
}
OpenNet_PacketInfo;
