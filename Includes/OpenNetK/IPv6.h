
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 1029 KMS. All rights reserved.
/// \file       Includes/OpenNetK/IPv6.h

// TEST COVERAGE  2019-05-03  KMS - Martin Dubois, ing.

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   The Ethernet type of IPv6 packets
/// \endcond
/// \cond    fr
/// \brief   Le type Ethernet des paquets IPv6
/// \endcond
#define IPv6_ETHERNET_TYPE_nh (0xdd86)

#ifndef _OPEN_NET_NO_FUNCTION_

// Functions
/////////////////////////////////////////////////////////////////////////////

// LIMITATION  IPv6.MultiHeader
//             Not supported

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
OPEN_NET_GLOBAL unsigned char * IPv6_Data(OPEN_NET_GLOBAL unsigned char * aData)
{
    return aData + 40;
}

/// \cond    en
/// \brief   This function returns a pointer to the destination address.
/// \param   aData  A pointer to the IPv6 header
/// \return  This function returns a pointer to the destination address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers l'adresse de
///          destination.
/// \param   aData  Un pointeur vers l'ent&ecirc;te IPv6
/// \return  Cette fonction retourne un pointeur vers l'adresse de
///          destination.
/// \endcond
OPEN_NET_GLOBAL unsigned char * IPv6_Destination(OPEN_NET_GLOBAL unsigned char * aData)
{
    return (aData + 24);
}

/// \cond    en
/// \brief   This function returns the protocol.
/// \param   aData  A pointer to the IPv6 header
/// \return  This function returns the protocol.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne le protocole.
/// \param   aData  Un pointeur vers l'ent&ecirc;te IPv6
/// \return  Cette fonction retourne le protocole.
/// \endcond
unsigned char IPv6_Protocol(OPEN_NET_GLOBAL unsigned char * aData)
{
    return aData[6];
}

/// \cond    en
/// \brief   This function returns a pointer to the source address.
/// \param   aData  A pointer to the IPv6 header
/// \return  This function returns a pointer to the source address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers l'adresse de provenance.
/// \param   aData  Un pointeur vers l'ent&ecirc;te IPv6
/// \return  Cette fonction retourne un pointeur vers l'adresse de provenance.
/// \endcond
OPEN_NET_GLOBAL unsigned char * IPv6_Source(OPEN_NET_GLOBAL unsigned char * aData)
{
    return (aData + 8);
}

#endif // ! _OPEN_NET_NO_FUNCTION_
