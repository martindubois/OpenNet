
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 1029 KMS. All rights reserved.
/// \file       Includes/OpenNetK/UDP.h

// TEST COVERAGE  2019-05-03  KMS - Martin Dubois, ing.

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   The IP protocol of UDP packets
/// \endcond
/// \cond    fr
/// \brief   Le protocole IP des paquets UDP
/// \endcond
#define UDP_IP_PROTOCOL (0x11)

// Functions
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   This function returns the destination port.
/// \param   aData  A pointer to the UDP header
/// \return  This function returns the destination port.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne le port de destination.
/// \param   aData  Un pointeur vers l'ent&ecirc;te UDP
/// \return  Cette fonction retourne le port de destination.
/// \endcond
unsigned short UDP_DestinationPort(OPEN_NET_GLOBAL unsigned char * aData)
{
    return (*((OPEN_NET_GLOBAL unsigned short *)(aData + 2)));
}

/// \cond    en
/// \brief   This function returns the source port.
/// \param   aData  A pointer to the UDP header
/// \return  This function returns the source port.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne le port de source.
/// \param   aData  Un pointeur vers l'ent&ecirc;te UDP
/// \return  Cette fonction retourne le port de source.
/// \endcond
unsigned short UDP_SourcePort(OPEN_NET_GLOBAL unsigned char * aData)
{
    return (*((OPEN_NET_GLOBAL unsigned short *)(aData + 0)));
}
