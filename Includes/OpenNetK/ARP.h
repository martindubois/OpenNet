
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/ARP.h
/// \brief      ARP_Destination, ARP_ETHERNET_TYPE_nh, ARP_Protocol,
///             ARP_Source (RT)

// CODE REVIEW    2019-07-16  KMS - Martin Dubois, ing.

// TEST COVERAGE  2019-05-03  KMS - Martin Dubois, ing.

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   The Ethernet type of ARP packets
/// \endcond
/// \cond    fr
/// \brief   Le type Ethernet des paquets ARP
/// \endcond
#define ARP_ETHERNET_TYPE_nh (0x0608)

#ifndef _OPEN_NET_NO_FUNCTION_

// Functions
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   This function returns a pointer to the requested address.
/// \param   aData  A pointer to the ARP header
/// \return  This function returns a pointer to the requested address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers l'adresse
///          questionn&eacute;e.
/// \param   aData  Un pointeur vers l'ent&ecirc;te ARP
/// \return  Cette fonction retourne un pointeur vers l'adresse
///          questionn&eacute;e.
/// \endcond
OPEN_NET_GLOBAL unsigned short * ARP_Destination(OPEN_NET_GLOBAL unsigned char * aData)
{
    return ((OPEN_NET_GLOBAL unsigned short *)(aData + 24));
}

/// \cond    en
/// \brief   This function returns the protocol.
/// \param   aData  A pointer to the ARP header
/// \return  This function returns the protocol.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne le protocole.
/// \param   aData  Un pointeur vers l'ent&ecirc;te ARP
/// \return  Cette fonction retourne le protocole.
/// \endcond
unsigned short ARP_Protocol(OPEN_NET_GLOBAL unsigned char * aData)
{
	return (*((OPEN_NET_GLOBAL unsigned short *)(aData + 2)));
}

/// \cond    en
/// \brief   This function returns a pointer to the source address.
/// \param   aData  A pointer to the ARP header
/// \return  This function returns a pointer to the source address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers l'adresse de
///          provenance.
/// \param   aData  Un pointeur vers l'ent&ecirc;te ARP
/// \return  Cette fonction retourne un pointeur vers l'adresse de
///          provenance.
/// \endcond
OPEN_NET_GLOBAL unsigned short * ARP_Source(OPEN_NET_GLOBAL unsigned char * aData)
{
    return ((OPEN_NET_GLOBAL unsigned short *)(aData + 14));
}

#endif // ! _OPEN_NET_NO_FUNCTION_
