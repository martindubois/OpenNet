
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Ethernet.h

#pragma once

// Constant
/////////////////////////////////////////////////////////////////////////////

#define ETHERNET_VLAN_TAG_ID_nh (0x8100)

// Functions
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   This function returns a pointer to the Ethernet payload.
/// \param   aBase        A pointer to the buffer
/// \param   aPacketInfo  A pointer to the information about the packet to
///                       process
/// \return  This function returns a pointer to the Etehrnet payload.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers la charge utile du
///          paquet Ethernet.
/// \param   aBase        Un pointeur vers le d&eacute;but du paquet
/// \param   aPacketInfo  Un pointeur vers l'information au sujet du paquet a
///          traiter
/// \return  Cette fonction retourne un pointeur vers la charge utile du
///          paquet Ethernet.
/// \endcond
OPEN_NET_GLOBAL unsigned char * Ethernet_Data( OPEN_NET_GLOBAL unsigned char * aBase, OPEN_NET_GLOBAL const OpenNet_PacketInfo * aPacketInfo)
{
    unsigned short lType_nh = *((OPEN_NET_GLOBAL unsigned short *)(aBase + aPacketInfo->mOffset_byte + 12));

    return (aBase + aPacketInfo.mOffset_byte + ((ETHERNET_VLAN_TAG_ID_nh == lType) ? 18 : 14));
}

/// \cond    en
/// \brief   This function returns Ethernet packet type.
/// \param   aBase        A pointer to the buffer
/// \param   aPacketInfo  A pointer to the information about the packet to
///                       process
/// \return  This function returns the Etehrnet packet type.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne le type du paquet Ethernet.
/// \param   aBase        Un pointeur vers le d&eacute;but du paquet
/// \param   aPacketInfo  Un pointeur vers l'information au sujet du paquet a
///          traiter
/// \return  Cette fonction retourne le type du paquet Ethernet.
/// \endcond
unsigned short Ethernet_Type(__global const unsigned char * aBase, __global const OpenNet_PacketInfo * aPacketInfo)
{
    unsigned short lType_nh = *((OPEN_NET_GLOBAL unsigned short *)(aBase + aPacketInfo->mOffset_byte + 12));

    return (ETHERNET_VLAN_TAG_ID_nh == lType_nh) ? (*((__global const unsigned short *)(aBase + aPacketInfo->mOffset_byte + 16))) : lType_nh;
}
