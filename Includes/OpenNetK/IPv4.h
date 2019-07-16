
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-1029 KMS. All rights reserved.
/// \file       Includes/OpenNetK/IPv4.h
/// \brief      IPv4_Data, IPv4_DataSize, IPv4_ETHERNET_TYPE_nh,
///             IPv4_Destination, IPv4_HeaderSize, IPv4_Protocol, IPv4_Source
///             (RT)

// CODE REVIEW    2019-07-16  KMS - Martin Dubois, ing.

// TEST COVERAGE  2019-05-10  KMS - Martin Dubois, ing.

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/ByteOrder.h>

// Constants
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   The Ethernet type of IPv4 packets
/// \endcond
/// \cond    fr
/// \brief   Le type Ethernet des paquets IPv4
/// \endcond
#define IPv4_ETHERNET_TYPE_nh (0x0008)

#ifndef _OPEN_NET_NO_FUNCTION_

// Functions
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   This function returns a pointer to the destination address.
/// \param   aData  A pointer to the IPv4 header
/// \return  This function returns a pointer to the destination address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers l'adresse de
///          destination.
/// \param   aData  Un pointeur vers l'ent&ecirc;te IPv4
/// \return  Cette fonction retourne un pointeur vers l'adresse de
///          destination.
/// \endcond
OPEN_NET_GLOBAL unsigned short * IPv4_Destination(OPEN_NET_GLOBAL unsigned char * aData)
{
    return ((OPEN_NET_GLOBAL unsigned short *)(aData + 16));
}

/// \cond    en
/// \brief   This function returns the size of the IP header.
/// \param   aData  A pointer to the IPv4 header
/// \return  This function returns the size of the IP header.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne la taille de l'ent&ecirc;te IP.
/// \param   aData  Un pointeur vers l'ent&ecirc;te IPv4
/// \return  Cette fonction retourne la taille de l'ent&ecirc;te IP.
/// \endcond
unsigned int IPv4_HeaderSize(OPEN_NET_GLOBAL unsigned char * aData)
{
    return ( (aData[0] & 0x0f) * 4 );
}

/// \cond    en
/// \brief   This function returns the protocol.
/// \param   aData  A pointer to the IPv4 header
/// \return  This function returns the protocol.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne le protocole.
/// \param   aData  Un pointeur vers l'ent&ecirc;te IPv4
/// \return  Cette fonction retourne le protocole.
/// \endcond
unsigned char IPv4_Protocol(OPEN_NET_GLOBAL unsigned char * aData)
{
    return aData[9];
}

/// \cond    en
/// \brief   This function returns a pointer to the source address.
/// \param   aData  A pointer to the IPv4 header
/// \return  This function returns a pointer to the source address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers l'adresse de provenance.
/// \param   aData  Un pointeur vers l'ent&ecirc;te IPv4
/// \return  Cette fonction retourne un pointeur vers l'adresse de provenance.
/// \endcond
OPEN_NET_GLOBAL unsigned short * IPv4_Source(OPEN_NET_GLOBAL unsigned char * aData)
{
    return ((OPEN_NET_GLOBAL unsigned short *)(aData + 12));
}

// --------------------------------------------------------------------------

/// \cond    en
/// \brief   This function returns a pointer to the payload.
/// \param   aData  A pointer to the IPv6 header
/// \return  This function returns a pointer to the payload.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers la charge utile.
/// \param   aData  Un pointeur vers l'ent&ecirc;te IPv6
/// \return  Cette fonction retourne un pointeur vers la charge utile.
/// \endcond
OPEN_NET_GLOBAL unsigned char * IPv4_Data(OPEN_NET_GLOBAL unsigned char * aData)
{
    return (aData + IPv4_HeaderSize(aData));
}

/// \cond    en
/// \brief   This function returns the size of the payload.
/// \param   aData  A pointer to the IPv4 header
/// \return  This function returns the size of the payload.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne la taille de la charge utile.
/// \param   aData  Un pointeur vers l'ent&ecirc;te IPv4
/// \return  Cette fonction retourne la taille de la charge utile.
/// \endcond
unsigned int IPv4_DataSize(OPEN_NET_GLOBAL unsigned char * aData)
{
    return (ByteOrder_Swap16(*((OPEN_NET_GLOBAL unsigned short *)(aData + 2))) - IPv4_HeaderSize(aData));
}

#endif // ! _OPEN_NET_NO_FUNCTION_
