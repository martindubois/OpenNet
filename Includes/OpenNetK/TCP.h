
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 1029 KMS. All rights reserved.
/// \file       Includes/OpenNetK/TCP.h

// TEST COVERAGE  2019-05-10  KMS - Martin Dubois, ing.

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   The size of the TCP header
/// \endcond
/// \cond    fr
/// \brief   La taille de l'ent&ecirc;te TCP
/// \endcond
#define TCP_HEADER_SIZE_byte (20)

/// \cond    en
/// \brief   The IP protocol of TCP packets
/// \endcond
/// \cond    fr
/// \brief   Le protocole IP des paquets TCP
/// \endcond
#define TCP_IP_PROTOCOL (0x06)

#ifndef _OPEN_NET_NO_FUNCTION_

// Functions
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   This function returns a pointer to the payload.
/// \param   aData  A pointer to the TCP header
/// \return  This function returns a pointer to the payload.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne l'adresse de la charge utile.
/// \param   aData  Un pointeur vers l'ent&ecirc;te TCP
/// \return  Cette fonction retourne l'adresse de la charge utile.
/// \endcond
OPEN_NET_DEVICE OPEN_NET_GLOBAL unsigned char * TCP_Data(OPEN_NET_GLOBAL unsigned char * aData)
{
    return (aData + TCP_HEADER_SIZE_byte);
}

/// \cond    en
/// \brief   This function returns the destination port.
/// \param   aData  A pointer to the TCP header
/// \return  This function returns the destination port.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne le port de destination.
/// \param   aData  Un pointeur vers l'ent&ecirc;te UDP
/// \return  Cette fonction retourne le port de destination.
/// \endcond
OPEN_NET_DEVICE unsigned short TCP_DestinationPort(OPEN_NET_GLOBAL const unsigned char * aData)
{
    return (*((OPEN_NET_GLOBAL const unsigned short *)(aData + 2)));
}

/// \cond    en
/// \brief   This function returns the source port.
/// \param   aData  A pointer to the TCP header
/// \return  This function returns the source port.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne le port de source.
/// \param   aData  Un pointeur vers l'ent&ecirc;te UDP
/// \return  Cette fonction retourne le port de source.
/// \endcond
OPEN_NET_DEVICE unsigned short TCP_SourcePort(OPEN_NET_GLOBAL const unsigned char * aData)
{
    return (*((OPEN_NET_GLOBAL const unsigned short *)(aData + 0)));
}

#endif // ! _OPEN_NET_NO_FUNCTION_
