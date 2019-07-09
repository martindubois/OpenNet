
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Ethernet.h

// TEST COVERAGE  2019-05-10  KMS - Martin Dubois, ing.

#pragma once

// Constant
/////////////////////////////////////////////////////////////////////////////

#define ETHERNET_VLAN_TAG_ID_nh (0x0081)

#ifndef _OPEN_NET_NO_FUNCTION_

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
OPEN_NET_DEVICE OPEN_NET_GLOBAL unsigned char * Ethernet_Data( OPEN_NET_GLOBAL unsigned char * aBase, OPEN_NET_GLOBAL const OpenNet_PacketInfo * aPacketInfo)
{
    unsigned short lType_nh = *((OPEN_NET_GLOBAL unsigned short *)(aBase + aPacketInfo->mOffset_byte + 12));

    return (aBase + aPacketInfo->mOffset_byte + ((ETHERNET_VLAN_TAG_ID_nh == lType_nh) ? 18 : 14));
}

/// \cond    en
/// \brief   This function returns the size of the Ethernet payload.
/// \param   aBase        A pointer to the buffer
/// \param   aPacketInfo  A pointer to the information about the packet to
///                       process
/// \return  This function returns the size of the Etehrnet payload.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne la taille de la charge utile du paquet
///          Ethernet.
/// \param   aBase        Un pointeur vers le d&eacute;but du paquet
/// \param   aPacketInfo  Un pointeur vers l'information au sujet du paquet a
///          traiter
/// \return  Cette fonction retourne la taille de la charge utile du paquet
///          Ethernet.
/// \endcond
OPEN_NET_DEVICE unsigned int Ethernet_DataSize(OPEN_NET_GLOBAL unsigned char * aBase, OPEN_NET_GLOBAL const OpenNet_PacketInfo * aPacketInfo)
{
    unsigned short lType_nh = *((OPEN_NET_GLOBAL unsigned short *)(aBase + aPacketInfo->mOffset_byte + 12));

    return (aPacketInfo->mSize_byte - ((ETHERNET_VLAN_TAG_ID_nh == lType_nh) ? 18 : 14));
}

/// \cond    en
/// \brief   This function returns a pointer to the Ethernet destination
///          address.
/// \param   aBase        A pointer to the buffer
/// \param   aPacketInfo  A pointer to the information about the packet to
///                       process
/// \return  This function returns a pointer to the Etehrnet destination
///          address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers l'adresse de
///          destination.
/// \param   aBase        Un pointeur vers le d&eacute;but du paquet
/// \param   aPacketInfo  Un pointeur vers l'information au sujet du paquet a
///          traiter
/// \return  Cette fonction retourne un pointeur vers l'adresse de
///          destination.
/// \endcond
OPEN_NET_DEVICE OPEN_NET_GLOBAL unsigned short * Ethernet_Destination(OPEN_NET_GLOBAL unsigned char * aBase, OPEN_NET_GLOBAL const OpenNet_PacketInfo * aPacketInfo)
{
	return ((OPEN_NET_GLOBAL unsigned short *)(aBase + aPacketInfo->mOffset_byte + 0));
}

/// \cond    en
/// \brief   This function returns a pointer to the Ethernet source address.
/// \param   aBase        A pointer to the buffer
/// \param   aPacketInfo  A pointer to the information about the packet to
///                       process
/// \return  This function returns a pointer to the Etehrnet source address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers l'adresse de source.
/// \param   aBase        Un pointeur vers le d&eacute;but du paquet
/// \param   aPacketInfo  Un pointeur vers l'information au sujet du paquet a
///          traiter
/// \return  Cette fonction retourne un pointeur vers l'adresse de source.
/// \endcond
OPEN_NET_DEVICE OPEN_NET_GLOBAL unsigned short * Ethernet_Source(OPEN_NET_GLOBAL unsigned char * aBase, OPEN_NET_GLOBAL const OpenNet_PacketInfo * aPacketInfo)
{
	return ((OPEN_NET_GLOBAL unsigned short *)(aBase + aPacketInfo->mOffset_byte + 6));
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
OPEN_NET_DEVICE unsigned short Ethernet_Type(OPEN_NET_GLOBAL const unsigned char * aBase, OPEN_NET_GLOBAL const OpenNet_PacketInfo * aPacketInfo)
{
    unsigned short lType_nh = *((OPEN_NET_GLOBAL const unsigned short *)(aBase + aPacketInfo->mOffset_byte + 12));

    return (ETHERNET_VLAN_TAG_ID_nh == lType_nh) ? (*((OPEN_NET_GLOBAL const unsigned short *)(aBase + aPacketInfo->mOffset_byte + 16))) : lType_nh;
}

/// \cond    en
/// \brief   This function indicate if a VLAN tag is present.
/// \param   aBase        A pointer to the buffer
/// \param   aPacketInfo  A pointer to the information about the packet to
///                       process
/// \retval  0  No
/// \retval  1  Yes
/// \endcond
/// \cond    fr
/// \brief   Cette fonction indique si un tag VLAN est pr&eacute;sent.
/// \param   aBase        Un pointeur vers le d&eacute;but du paquet
/// \param   aPacketInfo  Un pointeur vers l'information au sujet du paquet a
///          traiter
/// \retval  0  Non
/// \retval  1  Oui
/// \endcond
OPEN_NET_DEVICE int Ethernet_Vlan(OPEN_NET_GLOBAL const unsigned char * aBase, OPEN_NET_GLOBAL const OpenNet_PacketInfo * aPacketInfo)
{
	unsigned short lType_nh = *((OPEN_NET_GLOBAL const unsigned short *)(aBase + aPacketInfo->mOffset_byte + 12));

	return (ETHERNET_VLAN_TAG_ID_nh == lType_nh);
}

/// \cond    en
/// \brief   This function returns the VLAN tag.
/// \param   aBase        A pointer to the buffer
/// \param   aPacketInfo  A pointer to the information about the packet to
///                       process
/// \return  This function returns 0 if the packet does not include a VLAN
///          tag.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne le tag VLAN.
/// \param   aBase        Un pointeur vers le d&eacute;but du paquet
/// \param   aPacketInfo  Un pointeur vers l'information au sujet du paquet a
///          traiter
/// \return  Cette fonction retourne 0 si le paquet ne contient pas de tag
///          VLAN.
/// \endcond
OPEN_NET_DEVICE unsigned short Ethernet_VlanTag(OPEN_NET_GLOBAL const unsigned char * aBase, OPEN_NET_GLOBAL const OpenNet_PacketInfo * aPacketInfo)
{
	unsigned short lType_nh = *((OPEN_NET_GLOBAL const unsigned short *)(aBase + aPacketInfo->mOffset_byte + 12));

	return (ETHERNET_VLAN_TAG_ID_nh == lType_nh) ? (*((OPEN_NET_GLOBAL const unsigned short *)(aBase + aPacketInfo->mOffset_byte + 14))) : 0;
}

#endif // ! _OPEN_NET_NO_FUNCTION_
