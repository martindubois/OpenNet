
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Types.h

#pragma once

// Constants / Constantes
/////////////////////////////////////////////////////////////////////////////

// ===== Buffer state =======================================================

// --> INVALID <--+<------------+<----------+
//        |       |             |           |
//        +--> RECEVING --> RECEIVED --> TO_SEND

#define OPEN_NET_BUFFER_STATE_INVALID   (0)
#define OPNE_NET_BUFFER_STATE_RECEIVED  (1)
#define OPEN_NET_BUFFER_STATE_RECEIVING (2)
#define OPEN_NET_BUFFER_STATE_TO_SEND   (3)
#define OPEN_NET_BUFFER_STATE_QTY       (4)

// ===== Packet state / Etat d'un paquet ====================================

// --> INVALID <--+<------------+<----------+
//        |       |             |           |
//        +--> RECEVING --> RECEIVED --> TO_SEND

#define OPEN_NET_PACKET_STATE_INVALID   (0)
#define OPNE_NET_PACKET_STATE_RECEIVED  (1)
#define OPEN_NET_PACKET_STATE_RECEIVING (2)
#define OPEN_NET_PACKET_STATE_TO_SEND   (3)
#define OPEN_NET_PACKET_STATE_QTY       (4)

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
    uint32_t mVersion;

    uint32_t mPacketCount          ;
    uint32_t mPacketInfoOffset_byte;
    uint32_t mPacketSize_byte      ;
    uint32_t mBufferState          ;

    uint8_t mReserved0[44];
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

    uint8_t mReserved[4];

    uint32_t mPacketOffset_byte;
    uint32_t mPacketState      ;
    uint32_t mPacketSize_byte  ;

    uint8_t mReserved0[12];
}
OpenNet_PacketInfo;
