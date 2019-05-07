
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-1029 KMS. All rights reserved.
/// \file       Includes/OpenNetK/IPv4.h

// TEST COVERAGE  2019-05-03  KMS - Martin Dubois, ing.

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   The Ethernet type of IPv4 packets
/// \endcond
/// \cond    fr
/// \brief   Le type Ethernet des paquets IPv4
/// \endcond
#define IPv4_ETHERNET_TYPE_nh (0x0008)

// Functions
/////////////////////////////////////////////////////////////////////////////

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
    unsigned char lHeaderLen = ( aData[0] & 0x0f ) * 4;

    return aData + lHeaderLen;
}

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
