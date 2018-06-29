
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Types.h

// TODO  OpenNetK/Types.h  Use enum for OPEN_NET_BUFFER_STATE_... and
//                         OPEN_NET_PACKET_STATE

// TODO  Includes.Types
//       Put volatile to the right place and only the right place

#pragma once

// Constants / Constantes
/////////////////////////////////////////////////////////////////////////////

// ===== Buffer state =======================================================

//                  +--> STOPPED <-----------------------+<---------------+
//                  |                                    |                |
// --> INVALID --> TX_RUNNING <-- TX_PROGRAMMING <----- PX_COMPLETED <--+ |
//                  |                                                   | |
//                  +--> RX_PROGRAMMING --> RX_RUNNING --> PX_RUNNING --+ |
//                                                                 |      |
//                                                                 +------+
#define OPEN_NET_BUFFER_STATE_INVALID        (0)
#define OPEN_NET_BUFFER_STATE_PX_COMPLETED   (1)
#define OPEN_NET_BUFFER_STATE_PX_RUNNING     (2)
#define OPEN_NET_BUFFER_STATE_RX_PROGRAMMING (3)
#define OPEN_NET_BUFFER_STATE_RX_RUNNING     (4)
#define OPEN_NET_BUFFER_STATE_STOPPED        (5)
#define OPEN_NET_BUFFER_STATE_TX_PROGRAMMING (6)
#define OPEN_NET_BUFFER_STATE_TX_RUNNING     (7)
#define OPEN_NET_BUFFER_STATE_QTY            (8)

// ===== Packet state / Etat d'un paquet ====================================

// --> INVALID --> TX_RUNNING <------------------------ PX_COMPLETED <--+
//                  |                                                   |
//                  +-------------> RX_RUNNING --> RX_COMPLETED --------+
#define OPEN_NET_PACKET_STATE_INVALID      (0)
#define OPEN_NET_PACKET_STATE_PX_COMPLETED (1)
#define OPEN_NET_PACKET_STATE_RX_COMPLETED (2)
#define OPEN_NET_PACKET_STATE_RX_RUNNING   (3)
#define OPEN_NET_PACKET_STATE_TX_RUNNING   (4)
#define OPEN_NET_PACKET_STATE_QTY          (5)

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

    volatile uint32_t mBufferState;

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
    volatile uint32_t mToSendTo;

    uint8_t mReserved0[4];

    uint32_t mPacketOffset_byte;

    volatile uint32_t mPacketState    ;
    volatile uint32_t mPacketSize_byte;

    uint8_t mReserved1[12];
}
OpenNet_PacketInfo;
